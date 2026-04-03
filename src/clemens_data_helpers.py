from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np

SPREADSHEET_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")

CONDITION_LABELS = {
    "wt": "WT",
    "mud": "mudmut",
    "nanobody": "nanobody",
}


@dataclass
class ClemensCellCountRecord:
    condition: str
    lobe: str
    lineage: str
    total_cell_count: float


@dataclass
class ClemensVolumeRecord:
    condition: str
    lobe: str
    lineage: str
    total_cell_count: float
    lineage_volume_um3: float
    lineage_projected_area_um2: float
    avg_neuroblast_volume_um3: float
    avg_neuroblast_projected_area_um2: float
    n_dpn: int


def maybe_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return float("nan")
    return float(text)


def sphere_projected_area_from_volume(volume_um3: float) -> float:
    radius = ((3.0 * volume_um3) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return math.pi * radius * radius


def excel_column_index(col_letters: str) -> int:
    idx = 0
    for char in col_letters:
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx


def extract_cell_value(cell: ET.Element) -> str | float | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("x:is", SPREADSHEET_NS)
        if inline is None:
            return None
        return "".join(inline.itertext()).strip()

    value_node = cell.find("x:v", SPREADSHEET_NS)
    if value_node is None or value_node.text is None:
        return None

    raw_value = value_node.text.strip()
    if raw_value == "":
        return None

    try:
        numeric_value = float(raw_value)
    except ValueError:
        return raw_value

    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def load_worksheet(path: Path) -> list[list[str | float | None]]:
    with ZipFile(path) as workbook_zip:
        sheet_xml = workbook_zip.read("xl/worksheets/sheet1.xml")

    root = ET.fromstring(sheet_xml)
    sheet_data = root.find("x:sheetData", SPREADSHEET_NS)
    if sheet_data is None:
        raise ValueError(f"No sheet data found in {path}")

    rows: list[list[str | float | None]] = []
    for row in sheet_data.findall("x:row", SPREADSHEET_NS):
        row_values: list[str | float | None] = []
        for cell in row.findall("x:c", SPREADSHEET_NS):
            ref = cell.attrib.get("r", "")
            match = CELL_REF_RE.fullmatch(ref)
            if match is None:
                continue
            col_idx = excel_column_index(match.group(1)) - 1
            while len(row_values) <= col_idx:
                row_values.append(None)
            row_values[col_idx] = extract_cell_value(cell)
        rows.append(row_values)

    return rows


def normalize_lobe_name(raw_lobe: str | float | None) -> str:
    if raw_lobe is None:
        return ""
    text = str(raw_lobe).strip().lower()
    match = re.search(r"(\d+)$", text)
    if match is None:
        return text
    digits = match.group(1)
    if len(digits) <= 2:
        digits = digits.zfill(2)
    return f"lobe{digits}"


def infer_condition_key(path: Path) -> str:
    stem = path.stem.lower()
    for key in CONDITION_LABELS:
        if key in stem:
            return key
    raise ValueError(f"Could not infer condition from {path.name}")


def load_clemens_records(clemens_dir: Path) -> list[ClemensCellCountRecord]:
    records: list[ClemensCellCountRecord] = []

    for workbook_path in sorted(clemens_dir.glob("Pros_cell_counts*.xlsx")):
        condition_key = infer_condition_key(workbook_path)
        condition_label = CONDITION_LABELS[condition_key]
        rows = load_worksheet(workbook_path)
        if not rows:
            continue

        header = rows[0]
        lineage_names = [str(value).strip() if value is not None else "" for value in header[1:]]

        for row in rows[1:]:
            if not row:
                continue
            lobe = row[0]
            if lobe is None or str(lobe).strip() == "":
                continue

            lobe_name = normalize_lobe_name(lobe)
            for lineage_name, value in zip(lineage_names, row[1:]):
                if lineage_name == "" or value is None:
                    continue
                numeric_value = float(value)
                if numeric_value == 0:
                    continue
                records.append(
                    ClemensCellCountRecord(
                        condition=condition_label,
                        lobe=lobe_name,
                        lineage=lineage_name,
                        total_cell_count=numeric_value,
                    )
                )

    return records


def load_workbook_shared_strings(workbook_path: Path) -> list[str]:
    with ZipFile(workbook_path) as workbook_zip:
        root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
    return ["".join(si.itertext()) for si in root.findall("x:si", SPREADSHEET_NS)]


def load_workbook_sheet_targets(workbook_path: Path) -> dict[str, str]:
    with ZipFile(workbook_path) as workbook_zip:
        workbook_root = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
        rels_root = ET.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))

    rel_target_by_id = {
        rel.attrib["Id"]: "xl/" + rel.attrib["Target"].lstrip("/")
        for rel in rels_root
    }
    return {
        sheet.attrib["name"]: rel_target_by_id[
            sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
        ]
        for sheet in workbook_root.find("x:sheets", SPREADSHEET_NS)
    }


def load_worksheet_from_workbook(
    workbook_path: Path,
    worksheet_target: str,
    shared_strings: list[str],
) -> list[list[str | float | None]]:
    with ZipFile(workbook_path) as workbook_zip:
        sheet_xml = workbook_zip.read(worksheet_target)

    root = ET.fromstring(sheet_xml)
    sheet_data = root.find("x:sheetData", SPREADSHEET_NS)
    if sheet_data is None:
        raise ValueError(f"No sheet data found in {workbook_path}:{worksheet_target}")

    rows: list[list[str | float | None]] = []
    for row in sheet_data.findall("x:row", SPREADSHEET_NS):
        row_values: list[str | float | None] = []
        for cell in row.findall("x:c", SPREADSHEET_NS):
            ref = cell.attrib.get("r", "")
            match = CELL_REF_RE.fullmatch(ref)
            if match is None:
                continue
            col_idx = excel_column_index(match.group(1)) - 1
            while len(row_values) <= col_idx:
                row_values.append(None)

            cell_type = cell.attrib.get("t")
            if cell_type == "s":
                value_node = cell.find("x:v", SPREADSHEET_NS)
                if value_node is None or value_node.text is None:
                    row_values[col_idx] = None
                else:
                    row_values[col_idx] = shared_strings[int(value_node.text)]
            else:
                row_values[col_idx] = extract_cell_value(cell)
        rows.append(row_values)

    return rows


def build_cell_count_lookup(
    records: list[ClemensCellCountRecord],
) -> dict[tuple[str, str, str], float]:
    return {
        (record.condition, record.lobe, record.lineage): record.total_cell_count
        for record in records
    }


def load_clemens_volume_records(
    clemens_dir: Path,
    cell_count_lookup: dict[tuple[str, str, str], float],
) -> list[ClemensVolumeRecord]:
    workbook_path = clemens_dir / "Figure4_raw_data_classified_lineages.xlsx"
    shared_strings = load_workbook_shared_strings(workbook_path)
    sheet_targets = load_workbook_sheet_targets(workbook_path)

    volume_records: list[ClemensVolumeRecord] = []
    condition_to_sheet_suffix = {
        "WT": "wt",
        "nanobody": "nanobody",
        "mudmut": "mud",
    }

    for condition_label, sheet_suffix in condition_to_sheet_suffix.items():
        lineage_rows = load_worksheet_from_workbook(
            workbook_path,
            sheet_targets[f"Lineage_{sheet_suffix}"],
            shared_strings,
        )
        dpn_rows = load_worksheet_from_workbook(
            workbook_path,
            sheet_targets[f"Dpn_{sheet_suffix}"],
            shared_strings,
        )

        lineage_header = lineage_rows[0]
        lineage_lobe_idx = lineage_header.index("Lobe/File")
        lineage_class_idx = lineage_header.index("Set 1")
        lineage_volume_idx = lineage_header.index("Volume")

        lineage_volume_by_key: dict[tuple[str, str], float] = {}
        for row in lineage_rows[1:]:
            if len(row) <= max(lineage_lobe_idx, lineage_class_idx, lineage_volume_idx):
                continue
            raw_lobe = row[lineage_lobe_idx]
            lineage_name = row[lineage_class_idx]
            lineage_volume = row[lineage_volume_idx]
            if raw_lobe is None or lineage_name is None or lineage_volume is None:
                continue
            key = (normalize_lobe_name(raw_lobe), str(lineage_name).strip())
            lineage_volume_by_key[key] = float(lineage_volume)

        dpn_header = dpn_rows[0]
        dpn_lobe_idx = dpn_header.index("Lobe/File")
        dpn_class_idx = dpn_header.index("Set 1")
        dpn_volume_idx = dpn_header.index("Volume")

        dpn_volumes_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in dpn_rows[1:]:
            if len(row) <= max(dpn_lobe_idx, dpn_class_idx, dpn_volume_idx):
                continue
            raw_lobe = row[dpn_lobe_idx]
            lineage_name = row[dpn_class_idx]
            dpn_volume = row[dpn_volume_idx]
            if raw_lobe is None or lineage_name is None or dpn_volume is None:
                continue
            key = (normalize_lobe_name(raw_lobe), str(lineage_name).strip())
            dpn_volumes_by_key[key].append(float(dpn_volume))

        shared_keys = sorted(set(lineage_volume_by_key) & set(dpn_volumes_by_key))
        for lobe, lineage in shared_keys:
            lineage_volume = lineage_volume_by_key[(lobe, lineage)]
            dpn_volumes = dpn_volumes_by_key[(lobe, lineage)]
            avg_dpn_volume = float(np.mean(dpn_volumes))
            total_cell_count = cell_count_lookup.get(
                (condition_label, lobe, lineage), float("nan")
            )
            volume_records.append(
                ClemensVolumeRecord(
                    condition=condition_label,
                    lobe=lobe,
                    lineage=lineage,
                    total_cell_count=total_cell_count,
                    lineage_volume_um3=lineage_volume,
                    lineage_projected_area_um2=sphere_projected_area_from_volume(lineage_volume),
                    avg_neuroblast_volume_um3=avg_dpn_volume,
                    avg_neuroblast_projected_area_um2=sphere_projected_area_from_volume(avg_dpn_volume),
                    n_dpn=len(dpn_volumes),
                )
            )

    return volume_records
