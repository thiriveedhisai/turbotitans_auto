"""Printable AprilTag 36h11 zone markers + placement sheet for the real track.

    python3 markers/make_markers.py            (needs OpenCV with aruco, matplotlib, PyYAML;
                                                the map also needs src/tt_course_sim on PYTHONPATH)

IDs/zones come from config/tuning.yaml (marker_detector.tag_ids/tag_zones), so the prints
always match what the detector expects. Output (next to this script):
  png/tag36h11_<id>_<zone>.png  bare tag, black square only (1 px = 1 cell / 100)
  pdf/<id>_<zone>.pdf           A3 page, black square exactly TAG_MM wide + 100 mm check bar
  all_markers.pdf               every page in one file
  placement.csv                 where to stick each tag
  placement_map.png             course top view, tag ids + approach direction
"""
import csv
import math
import os
import sys

import cv2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TUNING = os.path.join(ROOT, 'src', 'tt_obstacle_course', 'config', 'tuning.yaml')
TAG_MM = 200.0                       # == marker_detector.tag_size (0.20 m)
SPEED_COURSE = [(12, 'speed_1'), (13, 'speed_2'), (14, 'speed_3'), (15, 'speed_4')]

# Obstacle-course placement (mirrors tt_course_sim/course.py, which the 3-D sim is run on).
# All: panel centre 0.50 m above the floor (just above the 0.36 m bales), face square to the
# lane, facing the truck as it arrives.
WHERE = {
    'start': 'Start/finish line, on the right-hand (north) lane wall; the truck starts beside it.',
    'ramp_up': 'Foot of the ramp (3.33 m past the start line), right-hand wall line.',
    'tunnel': 'Helix exit, where the 32" funnel into the tunnel starts; right-hand (east) wall.',
    'narrow': 'End of the tunnel straight, where the lane narrows 32" -> 20"; right-hand (east) wall.',
    'gravel': 'Foot of the gravel ramp-up (start of the 32" gravel entry); right-hand (south) wall.',
    'bank_approach': 'Zone line at the start of the bank entry lane; right-hand (south) side, '
                     '5 cm outside the lane wall.',
    'pothole': 'Zone line at the start of the return lane (west of the bank); right-hand (north) wall.',
    'wide': 'Where the pothole lane ends into the wide area; right-hand (north) side, 0.66 m off '
            'the lane centre line. Visible from inside the pothole lane, so it only confirms.',
    'buckets': 'On the wide-area wall that holds the bucket-field entrance, 0.5 m to the right '
               '(south) of the entrance gap centre.',
    'hoops': 'On the bucket-field far wall, 0.5 m to the right (south) of the exit gap centre '
             '(start of the 36" connector).',
    'car_wash': 'East end of the car-wash lane (lane mouth after the hoop U-turn), right-hand '
                '(north) side, 0.25 m outside the lane wall.',
    'finish': '0.85 m before the finish line inside the 32" funnel, right-hand (north) side.',
}


def tag_image(tag_id, cell_px=100):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    gen = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
    return gen(d, int(tag_id), 8 * cell_px)          # 6x6 data + 1-cell black border


def page(tag_id, zone, img, note):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mm = 1 / 25.4
    W, H = 297.0, 420.0                  # A3: a 200 mm tag needs ~1 cell (25 mm) of white around it
    fig = plt.figure(figsize=(W * mm, H * mm))
    left, bottom = (W - TAG_MM) / 2, 110.0
    ax = fig.add_axes([left / W, bottom / H, TAG_MM / W, TAG_MM / H])
    # Vector cells (crisp at any printer resolution; an embedded bitmap came out garbled).
    n = 8
    cell = img.shape[0] // n
    for r in range(n):
        for c in range(n):
            if img[r * cell + cell // 2, c * cell + cell // 2] < 128:
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1, color='k', lw=0))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_axis_off()
    fig.text(0.5, (bottom + TAG_MM + 50) / H, f'{zone}', ha='center', fontsize=40, weight='bold')
    fig.text(0.5, (bottom + TAG_MM + 38) / H, f'AprilTag 36h11  ID {tag_id}', ha='center', fontsize=18)
    fig.text(0.5, 80 / H, f'Black square must measure {TAG_MM:.0f} mm. Print A3 at 100% / actual size '
             'on matte paper; keep at least 25 mm of white around the black square.',
             ha='center', fontsize=9)
    fig.text(0.5, 70 / H, note, ha='center', fontsize=9, wrap=True)
    # 100 mm check bar.
    bar = fig.add_axes([(W - 100) / 2 / W, 40 / H, 100 / W, 2 / H])
    bar.add_patch(plt.Rectangle((0, 0), 1, 1, color='k'))
    bar.set_axis_off()
    fig.text(0.5, 33 / H, '|<-- this bar is 100 mm -->|', ha='center', fontsize=8)
    return fig


def course_positions():
    sys.path.insert(0, os.path.join(ROOT, 'src', 'tt_course_sim'))
    try:
        from tt_course_sim.course import Course, preview
    except ImportError:
        return None, None
    c = Course(rails=True, seed=0)
    return c, preview


def main():
    from matplotlib.backends.backend_pdf import PdfPages
    with open(TUNING) as f:
        md = yaml.safe_load(f)['marker_detector']['ros__parameters']
    tags = list(zip(md['tag_ids'], md['tag_zones']))
    os.makedirs(os.path.join(HERE, 'png'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'pdf'), exist_ok=True)
    course, preview = course_positions()
    pos = {}
    if course is not None:
        for (zone, x, y), yaw in zip(course.markers, course.marker_yaws):
            pos[zone] = (x, y, math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))))

    rows = []
    with PdfPages(os.path.join(HERE, 'all_markers.pdf')) as pdf:
        for tag_id, zone in tags + SPEED_COURSE:
            img = tag_image(tag_id)
            cv2.imwrite(os.path.join(HERE, 'png', f'tag36h11_{tag_id:02d}_{zone}.png'), img)
            speed = zone.startswith('speed_')
            where = ('Speed course (reserved: not in marker_detector.tag_ids, so the obstacle '
                     'mission ignores it).' if speed else WHERE.get(zone, ''))
            fig = page(tag_id, zone, img, where if len(where) < 120 else 'See placement.csv / placement_map.png')
            fig.savefig(os.path.join(HERE, 'pdf', f'{tag_id:02d}_{zone}.pdf'))
            pdf.savefig(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
            x, y, heading = pos.get(zone, ('', '', ''))
            rows.append({'id': tag_id, 'zone': zone, 'where': where,
                         'centre_height_m': '' if speed else 0.50,
                         'sim_x_m': '' if x == '' else round(x, 2),
                         'sim_y_m': '' if y == '' else round(y, 2),
                         'faces_truck_travelling_deg': '' if heading == '' else round(heading)})
    with open(os.path.join(HERE, 'placement.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if course is not None:
        fig, ax = preview(None, course)
        ids = dict((z, i) for i, z in tags)
        for zone, (x, y, heading) in pos.items():
            a = math.radians(heading)
            ax.annotate('', xy=(x, y), xytext=(x - 0.7 * math.cos(a), y - 0.7 * math.sin(a)),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
            ax.annotate(f'ID {ids.get(zone, "?")}', (x, y), fontsize=9, color='blue', weight='bold',
                        xytext=(4, -12), textcoords='offset points')
        ax.set_title('Zone markers (^ = tag, blue arrow = truck arriving; tag faces the arrow). '
                     'Seed-0 layout: buckets/hoops tags follow the real entrance/exit gaps.')
        fig.savefig(os.path.join(HERE, 'placement_map.png'), dpi=90, bbox_inches='tight')
    print(f'{len(rows)} markers -> {HERE}')


if __name__ == '__main__':
    main()
