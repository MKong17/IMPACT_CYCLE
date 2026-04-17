from __future__ import annotations

from typing import Dict, List, Tuple


def _parse_label_template(text: str) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            name, lid_text = line.rsplit(None, 1)
            rows.append((str(name).strip(), int(lid_text)))
        except Exception:
            continue
    return rows


_COARSE_TEMPLATE_TEXT = """
null 0
start_angle_grinder_assembly 1
unscrew_anti_vibration_handle 2
store_anti_vibration_handle 3
remove_locking_lever_assembly 4
store_locking_lever_assembly 5
extract_bearing_plate_assembly 6
store_bearing_plate_assembly 7
detach_adapter_plate 8
remove_rotor_assembly 9
store_rotor_assembly 10
store_gearbox_housing 11
store_adapter_plate 12
retrieve_gearbox_housing 13
retrieve_rotor_assembly 14
install_rotor_assembly 15
attach_adapter_plate 16
retrieve_adapter_plate 17
retrieve_anti_vibration_handle 18
retrieve_bearing_plate_assembly 19
insert_bearing_plate_assembly 20
retrieve_locking_lever_assembly 21
install_locking_lever_assembly 22
screw_on_anti_vibration_handle 23
store_tool 24
finish_angle_grinder_assembly 25
"""


_FINE_TEMPLATE_TEXT = """
null 0
pick_up_gearbox_housing 1
pick_up_gearbox_housing_drive_shaft 2
pick_up_drive_shaft 3
pick_up_bevel_gear 4
pick_up_adapter_plate 5
pick_up_bearing_plate 6
pick_up_screw 7
pick_up_M6_nut 8
pick_up_nut 9
pick_up_spring 10
pick_up_lever 11
pick_up_washer 12
pick_up_anti_vibration_handle 13
pick_up_flat_head_screwdriver 14
pick_up_combination_wrench 15
pick_up_phillips_screwdriver 16
pick_up_torx_screwdriver 17
place_gearbox_housing 18
place_gearbox_housing_drive_shaft 19
place_drive_shaft 20
place_bevel_gear 21
place_adapter_plate 22
place_bearing_plate 23
place_screw 24
place_M6_nut 25
place_nut 26
place_spring 27
place_lever 28
place_washer 29
place_anti_vibration_handle 30
place_flat_head_screwdriver 31
place_combination_wrench 32
place_phillips_screwdriver 33
place_torx_screwdriver 34
mount_adapter_plate 35
mount_anti_vibration_handle 36
insert_bevel_gear 37
insert_drive_shaft 38
insert_screw 39
attach_spring 40
attach_lever 41
seat_bearing_plate 42
dismount_gearbox_housing 43
dismount_adapter_plate 44
dismount_anti_vibration_handle 45
extract_drive_shaft 46
extract_bevel_gear 47
extract_bearing_plate 48
detach_spring 49
detach_lever 50
insert_M6_nut 51
thread_nut 52
thread_M6_nut 53
hand_tighten_nut 54
tighten_nut 55
hand_tighten_M6_nut 56
tighten_M6_nut 57
hand_tighten_screw 58
tighten_screw 59
hand_tighten_anti_vibration_handle 60
hand_loosen_nut 61
hand_loosen_M6_nut 62
loosen_nut 63
loosen_M6_nut 64
hand_loosen_screw 65
loosen_screw 66
hand_loosen_anti_vibration_handle 67
remove_screw 68
remove_washer 69
remove_nut 70
remove_M6_nut 71
adjust_drive_shaft 72
adjust_bearing_plate 73
align_lever 74
adjust_anti_vibration_handle 75
align_M6_nut 76
align_nut 77
align_screw 78
adjust_phillips_screwdriver 79
attach_screw 80
align_tool 81
transfer_gearbox_housing 82
transfer_gearbox_housing_drive_shaft 83
transfer_drive_shaft 84
transfer_bevel_gear 85
transfer_adapter_plate 86
transfer_bearing_plate 87
transfer_screw 88
transfer_nut 89
transfer_spring 90
transfer_lever 91
transfer_washer 92
transfer_anti_vibration_handle 93
transfer_flat_head_screwdriver 94
transfer_combination_wrench 95
transfer_phillips_screwdriver 96
transfer_torx_screwdriver 97
hold_gearbox_housing 98
hold_gearbox_housing_drive_shaft 99
hold_drive_shaft 100
hold_bevel_gear 101
hold_adapter_plate 102
hold_bearing_plate 103
hold_screw 104
hold_M6_nut 105
hold_nut 106
hold_spring 107
hold_lever 108
hold_washer 109
hold_anti_vibration_handle 110
hold_flat_head_screwdriver 111
hold_combination_wrench 112
hold_phillips_screwdriver 113
hold_torx_screwdriver 114
store_gearbox_housing 115
store_gearbox_housing_drive_shaft 116
store_drive_shaft 117
store_bevel_gear 118
store_adapter_plate 119
store_bearing_plate 120
store_screw 121
store_M6_nut 122
store_nut 123
store_spring 124
store_lever 125
store_washer 126
store_anti_vibration_handle 127
store_flat_head_screwdriver 128
store_combination_wrench 129
store_phillips_screwdriver 130
store_torx_screwdriver 131
flip_gearbox_housing 132
flip_gearbox_housing_drive_shaft 133
flip_adapter_plate 134
flip_bearing_plate 135
flip_lever 136
hand_spin_drive_shaft 137
"""


DEFAULT_ACTION_LABEL_TEMPLATES: Dict[str, List[Tuple[str, int]]] = {
    "Coarse": _parse_label_template(_COARSE_TEMPLATE_TEXT),
    "Fine": _parse_label_template(_FINE_TEMPLATE_TEXT),
}


def get_default_action_label_template(mode: str) -> List[Tuple[str, int]]:
    key = "Fine" if str(mode or "").strip().lower() == "fine" else "Coarse"
    return list(DEFAULT_ACTION_LABEL_TEMPLATES.get(key, DEFAULT_ACTION_LABEL_TEMPLATES["Coarse"]))
