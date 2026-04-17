ALGO_CONFIG = {
    "timeline_snap": {
        "playhead_radius": 6,
        "empty_space_radius": 10,
        "edge_search_radius": 5,
        "segment_soft_radius": 10,
        "phase_soft_radius": 8,
        "hover_preview_multi": True,
        "hover_preview_align": "absolute",
    },
    "boundary_snap": {
        "enabled": True,
        "window_size": 15,
    },
    "segment_embedding": {
        "trim_ratio": 0.1,
    },
    "topk": {
        "enabled": True,
        "k": 5,
        "uncertainty_margin": 0.25,
    },
    "assisted": {
        "boundary_min_gap": 15,
    },
}
