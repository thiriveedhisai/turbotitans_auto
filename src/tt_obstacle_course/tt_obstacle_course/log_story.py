"""The run as one uniform decision timeline, from any of the stack's logs. No ROS needed.

  tt_log ~/runs/0612_1405.log                  decisions of a finished (or partial) run
  tt_log -f ~/runs/0612_1405.log               follow a run that is still going
  ros2 launch ... 2>&1 | tee run.log | tt_log -    straight from the launch
  tt_log --zone buckets --all run.log          one zone, every line
  tt_log --summary run.log                     per-zone table: time, steps, stucks, tags
  tt_log --grep 'gap|exit' run.log             only lines matching a regex

Reads `ros2 launch` output, `ros2 run` output and tt_dash .jsonl recordings. Every line is
tagged with the zone and step the mission was in, and a category:
  START ZONE STEP BEHAV STUCK MARKER SEE TERRAIN ODOM RESULT CRASH LOG
"""

import argparse
import json
import re
import signal
import sys
import time

LINE = re.compile(r'^(?:\[(?P<proc>[\w.-]+)\]\s+)?\[(?P<lvl>DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\]\s+'
                  r'(?:\[(?P<t>\d+\.\d+)\]\s+)?\[(?P<node>[^\]]+)\]:\s?(?P<msg>.*)$')
PROC = re.compile(r'^\[(?P<proc>[\w.-]+)\]\s?(?P<msg>.*)$')
LEVELS = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}

# First match wins.
RULES = [(c, re.compile(p, re.I)) for c, p in [
    ('CRASH', r'process has died|Traceback|Exception|exit code [1-9]'),
    ('RESULT', r'=== 3D RUN|(?-i:FINISHED|FLIPPED)|HOOP:|RUN (STUCK|TIMEOUT)'),
    ('START', r'RACE START|GREEN at|Armed|E-STOP|Mission loaded|turning radius'),
    ('ZONE', r'=== ZONE|jumping from|zone \w+ timed out'),
    ('STEP', r'^step \d+'),
    ('STUCK', r'stuck|backing off|fallback|failed|giving up|no gate found|not known'),
    ('BEHAV', r'behavior <-|-> (done|running)'),
    ('MARKER', r'marker'),
    ('MAP', r'^map[: ]|map frame|map localization|teleop'),
    ('ODOM', r'tracking|odom|grid phase|drift|dead-reckon'),
    ('SEE', r'gap|exit|gate|hoop|bank plane|lane|wall|bales|light'),
    ('TERRAIN', r'terrain|lidar|tilt|calibrated'),
]]
CATEGORIES = ['START', 'ZONE', 'STEP', 'BEHAV', 'STUCK', 'MARKER', 'MAP', 'SEE', 'TERRAIN', 'ODOM',
              'RESULT', 'CRASH', 'LOG']
# Repeating perception chatter: shown with --all only.
NOISE = re.compile(r'bright exit at|gate rejected|tracking (OK|LOST) \(|Creating .* Bridge|'
                   r'pci id|libEGL|Warning \[Utils')
QUIET_NODES = {'gz', 'parameter_bridge', 'gz_bridge', 'rclcpp', 'static_transform_publisher',
               'laser_tf', 'launch', 'robot_state_publisher', 'tt_dash'}
COLOURS = {'START': '1;32', 'ZONE': '1;36', 'STEP': '36', 'BEHAV': '34', 'STUCK': '1;31',
           'MARKER': '35', 'MAP': '1;33', 'SEE': '33', 'TERRAIN': '32', 'ODOM': '90', 'RESULT': '1;37;44',
           'CRASH': '1;37;41', 'LOG': '0'}


def classify(msg, lvl='INFO'):
    for cat, pat in RULES:
        if pat.search(msg):
            return cat
    return 'STUCK' if lvl in ('ERROR', 'FATAL') else 'LOG'


def parse_line(line):
    """A launch/run log line -> event dict (t may be None), or None for blank lines."""
    line = line.rstrip('\n')
    if not line.strip():
        return None
    m = LINE.match(line)
    if m:
        lvl = 'WARN' if m['lvl'] == 'WARNING' else m['lvl']
        node = m['node'].split('.')[-1]
        return {'t': float(m['t']) if m['t'] else None, 'node': node, 'lvl': lvl,
                'msg': m['msg']}
    m = PROC.match(line)
    if m:
        return {'t': None, 'node': m['proc'].rsplit('-', 1)[0], 'lvl': 'INFO', 'msg': m['msg']}
    return {'t': None, 'node': '', 'lvl': 'INFO', 'msg': line}


class Story:
    """Tags each event with the mission's zone/step and the time since race start."""

    def __init__(self):
        self.zone = '-'
        self.step = '-'
        self.t0 = None
        self.last_t = None

    def feed(self, ev):
        if ev.get('t') is None:
            ev['t'] = self.last_t
        else:
            self.last_t = ev['t']
        msg = ev['msg']
        if self.t0 is None and ev['t'] is not None and (
                'RACE START' in msg or '=== ZONE' in msg):
            self.t0 = ev['t']
        m = re.search(r'=== ZONE (\w+)', msg)
        if m:
            self.zone, self.step = m[1], '-'
        m = re.match(r'step (\d+)', msg)
        if m and ev['node'] == 'mission_manager':
            self.step = m[1]
        ev.setdefault('cat', classify(msg, ev['lvl']))
        ev['zone'] = self.zone
        ev['step'] = self.step
        ev['el'] = None if self.t0 is None or ev['t'] is None else round(ev['t'] - self.t0, 2)
        ev['noise'] = bool(NOISE.search(msg)) or ev['node'] in QUIET_NODES
        return ev


def read_events(stream, follow=False):
    """Yield raw events from a text log or a tt_dash .jsonl, optionally following it."""
    while True:
        line = stream.readline()
        if not line:
            if not follow:
                return
            time.sleep(0.2)
            continue
        if line.startswith('{'):
            try:
                ev = json.loads(line)
                yield {k: ev.get(k) for k in ('t', 'node', 'lvl', 'msg')}
                continue
            except ValueError:
                pass
        ev = parse_line(line)
        if ev:
            yield ev


def fmt(ev, colour=False):
    el = f'{ev["el"]:7.1f}s' if ev.get('el') is not None else '      - '
    where = f'{ev["zone"]}/{ev["step"]}'
    text = f'{el} {where:>15} {ev["cat"]:7} {ev["node"][:18]:18} {ev["msg"]}'
    if colour:
        code = COLOURS.get(ev['cat'], '0')
        if ev['lvl'] in ('WARN',) and ev['cat'] == 'LOG':
            code = '33'
        return f'\033[{code}m{text}\033[0m'
    return text


class Summary:
    """Per-zone: when entered, how long, steps and how they ended, stucks, tags seen."""

    def __init__(self):
        self.zones = []

    def feed(self, ev):
        msg = ev['msg']
        if msg.startswith('=== ZONE') or not self.zones:
            if not msg.startswith('=== ZONE'):
                return
            self.zones.append({'zone': ev['zone'], 'in': ev.get('el'), 'steps': [],
                               'stuck': 0, 'tags': [], 'warn': 0, 'notes': []})
            return
        z = self.zones[-1]
        z['out'] = ev.get('el')
        m = re.match(r'step (\d+): (\w+)=?\{?\'?(\w+)?', msg)
        if m and ev['node'] == 'mission_manager':
            z['steps'].append([m[1], m[3] if m[2] in ('behavior',) else m[2], '?'])
        m = re.match(r'step (\d+) (ended: |skipped)(.*)', msg)
        if m and z['steps']:
            z['steps'][-1][2] = 'skipped' if m[2] == 'skipped' else m[3].strip()
        if 'backing off' in msg:
            z['stuck'] += 1
        m = re.search(r'marker -> zone (\w+)', msg)
        if m and m[1] not in z['tags']:
            z['tags'].append(m[1])
        if ev['lvl'] in ('WARN', 'ERROR') and not ev['noise']:
            z['warn'] += 1
        if ev['cat'] in ('RESULT', 'CRASH') or 'timed out' in msg or 'jumping' in msg:
            z['notes'].append(msg[:70])

    def text(self):
        out = [f'{"zone":14} {"in(s)":>7} {"dur(s)":>7} {"stuck":>5} {"warn":>4}  steps (i:what->end) | tags'
               '\n' + '-' * 110]
        for i, z in enumerate(self.zones):
            t_in = z['in']
            nxt = self.zones[i + 1]['in'] if i + 1 < len(self.zones) else z.get('out')
            dur = f'{nxt - t_in:7.1f}' if None not in (t_in, nxt) else '      -'
            steps = ' '.join(f'{s}:{w}->{e}' for s, w, e in z['steps'])
            tin = f'{t_in:7.1f}' if t_in is not None else '      -'
            out.append(f'{z["zone"]:14} {tin} {dur} {z["stuck"]:5} {z["warn"]:4}  {steps}'
                       f'{" | " + ",".join(z["tags"]) if z["tags"] else ""}')
            for n in z['notes']:
                out.append(f'{"":36}! {n}')
        return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 1)[1])
    ap.add_argument('file', help="launch log, tt_dash .jsonl, or '-' for stdin")
    ap.add_argument('-f', '--follow', action='store_true', help='keep reading as the file grows')
    ap.add_argument('--all', action='store_true', help='include chatter (bright exit, rejects, sim)')
    ap.add_argument('--zone', nargs='+', help='only these zones')
    ap.add_argument('--cat', nargs='+', choices=CATEGORIES, help='only these categories')
    ap.add_argument('--grep', help='only lines matching this regex')
    ap.add_argument('--summary', action='store_true', help='per-zone table at the end')
    ap.add_argument('--no-colour', action='store_true')
    a = ap.parse_args(argv)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    stream = sys.stdin if a.file == '-' else open(a.file, errors='replace')
    colour = sys.stdout.isatty() and not a.no_colour
    grep = re.compile(a.grep, re.I) if a.grep else None
    story, summary = Story(), Summary()
    try:
        for ev in read_events(stream, a.follow):
            story.feed(ev)
            summary.feed(ev)
            if a.summary and not a.follow:
                continue
            if ev['noise'] and not a.all:
                continue
            if not a.all and ev['cat'] == 'LOG' and ev['lvl'] == 'INFO':
                continue
            if a.zone and ev['zone'] not in a.zone:
                continue
            if a.cat and ev['cat'] not in a.cat:
                continue
            if grep and not grep.search(ev['msg']):
                continue
            print(fmt(ev, colour), flush=True)
    except KeyboardInterrupt:
        pass
    if a.summary:
        print(summary.text())


if __name__ == '__main__':
    main()
