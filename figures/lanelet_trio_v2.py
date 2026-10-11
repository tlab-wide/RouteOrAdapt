#!/usr/bin/env python3
"""Render Fig 5: M1/M2/M3 lanelet trio with paper-grade aesthetic.

Design highlights:
  - Unified slate-gray / blue-gray palette across all 3 panels.
  - Lane area filled as polygons (left bound + reversed right bound).
  - Aligned 100 m scale bar at bottom-left of every panel.
  - Card styling (#fafbfc background, #dde2e7 thin border).
  - Italic narrative subtitle baked into each panel.
  - Equal aspect, square-ish framing per panel.
"""
import os
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import PatchCollection

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
MAPS = [
    {
        "code": "Knowledge Hub",
        "name": "research campus",
        "subtitle": "paired bidirectional lanes",
        "osm": os.environ.get("MAPS_DIR", "maps") + "/" + "知の拠点あいち/milee_service_knowledge_hub_aich.osm",
    },
    {
        "code": "Ishiodai",
        "name": "suburban district",
        "subtitle": "paired bidirectional lanes",
        "osm": os.environ.get("MAPS_DIR", "maps") + "/" + "石尾台地区/lanelet2_map.osm",
    },
    {
        "code": "Morikoro",
        "name": "open park",
        "subtitle": "no paired-opposing lanes",
        "osm": os.environ.get("MAPS_DIR", "maps") + "/" + "モリコロ/モリコロ全体_20200413_LL2.osm",
    },
]

OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "fig5_lanelet_trio.pdf")
OUT_PNG = OUT_PDF.replace(".pdf", ".png")

# -----------------------------------------------------------------------------
# Unified palette
# -----------------------------------------------------------------------------
COLOR_BOUNDARY = "#52514e"   # dark slate-gray
COLOR_LANE_FILL = "#b7d3f6"  # light blue-gray
COLOR_LANE_ALPHA = 0.55
COLOR_CARD_BG = "#ffffff"    # off-white card
COLOR_CARD_EDGE = "#ffffff"  # neutral hairline border
COLOR_SCALE_BAR = "#52514e"  # near-black
COLOR_SUBTITLE = "#898781"   # muted gray for italic narrative

LANE_EDGE_LW = 0.3
SCALE_BAR_M = 100.0          # meters
SCALE_LW = 1.1


# -----------------------------------------------------------------------------
# OSM parsing (lanelet2 / VMB style: nodes carry local_x, local_y)
# -----------------------------------------------------------------------------
def parse_lanelet_osm(path):
    """Return (nodes_dict, ways_dict, lanelet_relations).

    nodes_dict[id] = (x, y)  (local_x, local_y; meters)
    ways_dict[id]  = [(x, y), ...]
    lanelet_relations = [(left_way_id, right_way_id), ...]
    """
    tree = ET.parse(path)
    root = tree.getroot()

    nodes = {}
    for n in root.findall("node"):
        nid = n.get("id")
        lx = ly = None
        for t in n.findall("tag"):
            k = t.get("k")
            if k == "local_x":
                lx = float(t.get("v"))
            elif k == "local_y":
                ly = float(t.get("v"))
        if lx is None or ly is None:
            # fall back to lat/lon if no local coords (unlikely here)
            lat = n.get("lat")
            lon = n.get("lon")
            if lat is not None and lon is not None:
                # Cheap equirectangular projection (meters-ish near origin lat)
                # Not used unless local_x missing.
                lat = float(lat)
                lon = float(lon)
                lx = lon * 111000.0 * np.cos(np.radians(lat))
                ly = lat * 111000.0
        if lx is not None and ly is not None:
            nodes[nid] = (lx, ly)

    ways = {}
    for w in root.findall("way"):
        wid = w.get("id")
        pts = []
        for nd in w.findall("nd"):
            ref = nd.get("ref")
            if ref in nodes:
                pts.append(nodes[ref])
        if len(pts) >= 2:
            ways[wid] = pts

    lanelets = []
    for r in root.findall("relation"):
        is_lanelet = False
        for t in r.findall("tag"):
            if t.get("k") == "type" and t.get("v") == "lanelet":
                is_lanelet = True
                break
        if not is_lanelet:
            continue
        left = right = None
        for m in r.findall("member"):
            if m.get("type") != "way":
                continue
            role = m.get("role")
            if role == "left":
                left = m.get("ref")
            elif role == "right":
                right = m.get("ref")
        if left in ways and right in ways:
            lanelets.append((left, right))
    return nodes, ways, lanelets


# -----------------------------------------------------------------------------
# Bounding-box helper -> square framing with padding
# -----------------------------------------------------------------------------
def square_bbox(xs, ys, pad_frac=0.04):
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    half *= (1.0 + pad_frac)
    return cx - half, cx + half, cy - half, cy + half


# -----------------------------------------------------------------------------
# Plot a single lanelet panel
# -----------------------------------------------------------------------------
def plot_panel(ax, osm_path, code, name, subtitle):
    nodes, ways, lanelets = parse_lanelet_osm(osm_path)

    # Build lane area polygons (left fwd + right reversed)
    polygons = []
    all_xs, all_ys = [], []
    for left_id, right_id in lanelets:
        L = ways[left_id]
        R = ways[right_id]
        ring = list(L) + list(reversed(R))
        polygons.append(MplPolygon(ring, closed=True))
        for (x, y) in ring:
            all_xs.append(x)
            all_ys.append(y)

    if not polygons:
        # fallback: just plot all ways
        for w in ways.values():
            xs = [p[0] for p in w]
            ys = [p[1] for p in w]
            ax.plot(xs, ys, color=COLOR_BOUNDARY, linewidth=LANE_EDGE_LW)
            all_xs.extend(xs)
            all_ys.extend(ys)
    else:
        pc = PatchCollection(
            polygons,
            facecolor=COLOR_LANE_FILL,
            edgecolor="none",
            alpha=COLOR_LANE_ALPHA,
            linewidths=0,
        )
        ax.add_collection(pc)

        # Thin dark boundary lines (left + right) on top
        for left_id, right_id in lanelets:
            for wid in (left_id, right_id):
                pts = ways[wid]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(
                    xs, ys,
                    color=COLOR_BOUNDARY,
                    linewidth=LANE_EDGE_LW,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )

    # Square framing
    xmin, xmax, ymin, ymax = square_bbox(all_xs, all_ys, pad_frac=0.05)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    # Card styling: hide default frame; draw our own
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(COLOR_CARD_BG)

    # Hairline border inside axes (in axes coordinates)
    border = Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor=COLOR_CARD_EDGE,
        linewidth=0.8,
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(border)

    # ----- Aligned scale bar (100 m), bottom-left, 5% margin -----
    span = xmax - xmin
    bar_len_data = SCALE_BAR_M  # in same data units (meters)
    # If 100 m is too small (>40% of span) or huge (<3% of span), pick sensible alt
    frac = bar_len_data / span
    if frac > 0.45:
        bar_len_data = 50.0
        label = "50 m"
    elif frac < 0.04:
        # use 500 m for very wide maps
        bar_len_data = 500.0
        label = "500 m"
    else:
        label = "100 m"

    margin_x = 0.09 * span
    margin_y = 0.22 * (ymax - ymin)
    x0 = xmin + margin_x
    y0 = ymin + margin_y
    x1 = x0 + bar_len_data
    ax.plot(
        [x0, x1], [y0, y0],
        color=COLOR_SCALE_BAR,
        linewidth=SCALE_LW,
        solid_capstyle="butt",
        zorder=20,
    )
    # End-caps
    cap = 0.012 * (ymax - ymin)
    ax.plot([x0, x0], [y0 - cap, y0 + cap],
            color=COLOR_SCALE_BAR, linewidth=SCALE_LW, zorder=20)
    ax.plot([x1, x1], [y0 - cap, y0 + cap],
            color=COLOR_SCALE_BAR, linewidth=SCALE_LW, zorder=20)
    ax.text(
        0.5 * (x0 + x1), y0 + 0.025 * (ymax - ymin),
        label,
        ha="center", va="bottom",
        fontsize=6.0, color=COLOR_SCALE_BAR, zorder=20,
    )

    # ----- Title + italic subtitle -----
    # We use ax.set_title for the bold code/name, and a manual italic subtitle below it
    title_text = code
    ax.set_title(
        title_text,
        fontsize=7.5,
        fontweight="bold",
        pad=11,
        color="#0b0b0b",
    )
    # Subtitle: italic, gray, just below the title (placed in axes coords)
    ax.text(
        0.5, 1.02, f"{name}, {subtitle}",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=6.3,
        color=COLOR_SUBTITLE,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import paperstyle as ps
    ps.setup()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(5.4, 2.05),
        constrained_layout=True,
    )
    try:
        fig.get_layout_engine().set(hspace=0.0, wspace=0.05)
    except Exception:
        pass

    for ax, m in zip(axes, MAPS):
        print(f"[fig5] rendering {m['code']} from {m['osm']}")
        plot_panel(ax, m["osm"], m["code"], m["name"], m["subtitle"])

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[fig5] wrote {OUT_PDF}")
    print(f"[fig5] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
