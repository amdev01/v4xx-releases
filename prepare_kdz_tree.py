#!/usr/bin/env python3
"""Prepare a kdz-tool extract tree for repacking a custom V410 KDZ.

kdz-tool extract/repack uses a 4K-sparse on-disk layout for ``0.<part>.img``:

  file_offset = (chunk.start_sector - part_start_sector) * 4096
  bytes       = chunk.data_size   # contiguous raw 512-byte-sector payload

Raw flash images from TWRP zips / lglaf (``.bin`` / ``.img``) are plain
512-byte-sector layouts and must be converted before ``kdz-tool repack``.

Typical V410 frankenstein flow:
  1. Start from extracted V41007h (DLL/dylib + bootloader partitions)
  2. Import V41010d ``system.img`` (and optionally boot/laf/...) from the zip
  3. Import live ``PrimaryGPT.bin`` from lglaf (fixed userdata/grow layout)
  4. Synthesize BackupGPT (or import lglaf's) and align metadata start_sector
  5. ``./build/kdz-tool repack <out_dir> V41010d_custom.kdz``

Example:
  python3 scripts/prepare_kdz_tree.py \\
      --base V41007h \\
      --out V41010d_kdz \\
      --zip V41010d_stock_installer.zip \\
      --import-zip system \\
      --primary-gpt /path/to/PrimaryGPT.bin \\
      --sw-version V41010d
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional


SECTOR = 512
SPARSE_SECTOR = 4096


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text())


def save_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=4) + "\n")


def part_chunks(meta: dict, part: str) -> List[dict]:
    chunks = meta["dz"]["parts"]["0"].get(part)
    if not chunks:
        raise SystemExit(f"partition '{part}' not in metadata.json")
    return chunks


def sparse_file_size(chunks: List[dict]) -> int:
    part_start = chunks[0]["part_start_sector"]
    last = chunks[-1]
    return (last["start_sector"] + last["sector_count"] - part_start) * SPARSE_SECTOR


def trim_trailing_zeros(buf: bytes, min_size: int = SECTOR) -> bytes:
    end = len(buf)
    while end > min_size and buf[end - 1] == 0:
        end -= 1
    # Keep sector alignment for cleaner DZ payloads
    if end % SECTOR:
        end = min(len(buf), ((end + SECTOR - 1) // SECTOR) * SECTOR)
    return buf[:end] if end else buf[:min_size]


def read_raw_slice(raw: BinaryIO, raw_size: int, offset: int, size: int) -> bytes:
    if offset >= raw_size:
        return b"\x00" * size
    raw.seek(offset)
    data = raw.read(min(size, raw_size - offset))
    if len(data) < size:
        data += b"\x00" * (size - len(data))
    return data


def write_sparse_from_raw(
    raw_path: Path,
    out_img: Path,
    chunks: List[dict],
    *,
    trim_zeros: bool = True,
    dense: bool = False,
) -> List[dict]:
    """Convert a raw 512B-sector image into kdz-tool 4K-sparse ``0.<part>.img``.

    Updates and returns chunk dicts with new ``data_size`` values.
    """
    raw_size = raw_path.stat().st_size
    part_start = chunks[0]["part_start_sector"]
    out_size = sparse_file_size(chunks)

    out_img.parent.mkdir(parents=True, exist_ok=True)
    # Create sparse file cheaply where the FS supports it
    with open(out_img, "wb") as out:
        out.truncate(out_size)

    updated: List[dict] = []
    with open(raw_path, "rb") as raw, open(out_img, "r+b") as out:
        for chunk in chunks:
            start = chunk["start_sector"]
            count = chunk["sector_count"]
            rel = start - part_start
            raw_off = rel * SECTOR
            file_off = rel * SPARSE_SECTOR
            max_bytes = count * SECTOR

            payload = read_raw_slice(raw, raw_size, raw_off, max_bytes)
            if dense:
                data = payload
            elif trim_zeros:
                data = trim_trailing_zeros(payload)
            else:
                # Preserve original data_size when possible
                want = min(chunk.get("data_size", max_bytes), max_bytes)
                data = payload[:want]

            out.seek(file_off)
            out.write(data)

            new_chunk = dict(chunk)
            new_chunk["data_size"] = len(data)
            # file_offset/file_size/hash are rewrite leftovers; repack recomputes
            new_chunk["file_size"] = 0
            new_chunk["hash"] = ""
            updated.append(new_chunk)

    return updated


def parse_gpt_backup_lba(primary_raw: bytes) -> Optional[int]:
    off = primary_raw.find(b"EFI PART")
    if off < 0:
        return None
    # backup_lba is at +32 within GPT header
    return struct.unpack_from("<Q", primary_raw, off + 32)[0]


def import_primary_gpt(raw_path: Path, out_img: Path, chunks: List[dict]) -> List[dict]:
    """Import LAF/raw PrimaryGPT into 4K-sparse image; pad/truncate to chunk size."""
    data = raw_path.read_bytes()
    # Accept either a short LAF dump or a full GPT window
    max_bytes = chunks[0]["sector_count"] * SECTOR
    if len(data) < 512 or b"EFI PART" not in data[: max(len(data), 4096)]:
        # Maybe already a kdz-tool sparse image — take leading max_bytes of
        # logical content by reading 4K-sparse as if raw was written densely at 0
        if b"EFI PART" in data[:4096] or data[512:520] == b"EFI PART":
            pass
        else:
            raise SystemExit(f"{raw_path}: no EFI PART signature found")

    # If source is kdz-tool sparse (huge) and EFI at 512, use first max_bytes
    if len(data) > max_bytes and data.find(b"EFI PART") == 512:
        data = data[:max_bytes]
    elif len(data) > max_bytes:
        data = data[:max_bytes]
    else:
        data = data + b"\x00" * (max_bytes - len(data))

    tmp = out_img.with_suffix(".raw.tmp")
    tmp.write_bytes(data)
    try:
        return write_sparse_from_raw(tmp, out_img, chunks, trim_zeros=True, dense=False)
    finally:
        tmp.unlink(missing_ok=True)


def synthesize_backup_from_primary(
    primary_raw: bytes, window_sectors: int = 1024
) -> bytes:
    """Build KDZ-style BackupGPT (window ending at backup_lba) from primary."""
    off = primary_raw.find(b"EFI PART")
    if off < 0:
        raise SystemExit("primary GPT missing EFI PART")

    header = bytearray(primary_raw[off : off + 92])
    backup_lba = struct.unpack_from("<Q", header, 32)[0]
    num_entries, entry_size = struct.unpack_from("<II", header, 80)
    entries_bytes = num_entries * entry_size
    # Primary entries start at LBA 2
    entries = primary_raw[2 * SECTOR : 2 * SECTOR + entries_bytes]
    if len(entries) < entries_bytes:
        entries += b"\x00" * (entries_bytes - len(entries))

    entry_sectors = max(32, (entries_bytes + SECTOR - 1) // SECTOR)
    part_entry_lba = backup_lba - entry_sectors
    window_start = backup_lba - window_sectors + 1
    if part_entry_lba < window_start:
        raise SystemExit("BackupGPT window too small for entry array")

    image = bytearray(window_sectors * SECTOR)
    ent_off = (part_entry_lba - window_start) * SECTOR
    image[ent_off : ent_off + len(entries)] = entries

    struct.pack_into("<Q", header, 24, backup_lba)  # current_lba
    struct.pack_into("<Q", header, 32, 1)  # backup points at primary
    struct.pack_into("<Q", header, 72, part_entry_lba)
    header_size = struct.unpack_from("<I", header, 12)[0]
    hdr = bytearray(header[:header_size])
    struct.pack_into("<I", hdr, 16, 0)
    import binascii

    crc = binascii.crc32(hdr) & 0xFFFFFFFF
    struct.pack_into("<I", hdr, 16, crc)
    image[(window_sectors - 1) * SECTOR : (window_sectors - 1) * SECTOR + len(hdr)] = hdr
    return bytes(image), backup_lba, window_start


def import_backup_gpt(
    out_img: Path,
    chunks: List[dict],
    *,
    primary_raw: Optional[bytes] = None,
    backup_raw_path: Optional[Path] = None,
) -> List[dict]:
    chunk = dict(chunks[0])
    if backup_raw_path is not None:
        data = backup_raw_path.read_bytes()
        if b"EFI PART" not in data:
            raise SystemExit(f"{backup_raw_path}: no EFI PART")
        # Prefer trailing/window content if a full dump was given
        if len(data) > chunk["sector_count"] * SECTOR:
            # keep last window_sectors * 512
            data = data[-(chunk["sector_count"] * SECTOR) :]
        backup_lba = None
        window_start = chunk["start_sector"]
    else:
        if primary_raw is None:
            raise SystemExit("need --primary-gpt or --backup-gpt")
        data, backup_lba, window_start = synthesize_backup_from_primary(
            primary_raw, window_sectors=chunk["sector_count"]
        )
        chunk["start_sector"] = window_start
        chunk["name"] = f"BackupGPT_{window_start}.bin"
        chunk["part_start_sector"] = window_start

    tmp = out_img.with_suffix(".raw.tmp")
    # Ensure exact window size
    want = chunk["sector_count"] * SECTOR
    if len(data) < want:
        data += b"\x00" * (want - len(data))
    else:
        data = data[:want]
    tmp.write_bytes(data)
    try:
        updated = write_sparse_from_raw(tmp, out_img, [chunk], trim_zeros=True)
    finally:
        tmp.unlink(missing_ok=True)

    if backup_lba is not None:
        print(
            f"  BackupGPT: backup_lba={backup_lba} window_start={window_start} "
            f"({chunk['sector_count']} sectors)"
        )
    return updated


def extract_from_zip(zip_path: Path, names: Iterable[str], dest_dir: Path) -> Dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        namelist = set(zf.namelist())
        for name in names:
            # Accept part name or explicit member
            candidates = [
                name,
                f"{name}.img",
                f"{name}.bin",
                f"{name}_stock.img",
                f"recovery_stock.img" if name == "recovery" else "",
            ]
            candidates = [c for c in candidates if c]
            member = next((c for c in candidates if c in namelist), None)
            if member is None:
                raise SystemExit(
                    f"{zip_path}: no member for '{name}' "
                    f"(tried {', '.join(candidates)})"
                )
            target = dest_dir / Path(member).name
            print(f"  unzip {member} -> {target}")
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            # Map logical part name
            part = name
            if part.endswith(".img") or part.endswith(".bin"):
                part = Path(part).stem.replace("_stock", "")
            out[part] = target
    return out


def copy_base_tree(base: Path, out: Path, *, force: bool, clone_mode: str) -> None:
    """Clone extract tree. Partition images default to hardlink, then symlink.

    Never fall back to full copies of multi‑GB sparse images unless
    ``clone_mode=copy`` (dangerous for disk space).
    """
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists (pass --force to overwrite)")
        shutil.rmtree(out)
    print(f"Cloning base tree {base} -> {out} (images: {clone_mode})")
    out.mkdir(parents=True)
    shutil.copy2(base / "metadata.json", out / "metadata.json")
    if (base / "components").is_dir():
        shutil.copytree(base / "components", out / "components")
    for img in sorted(base.glob("0.*.img")):
        dest = out / img.name
        if clone_mode == "copy":
            print(f"  copy {img.name} ({img.stat().st_size / 1024**3:.1f} GiB)")
            shutil.copy2(img, dest)
            continue
        if clone_mode == "symlink":
            dest.symlink_to(img.resolve())
            print(f"  symlink {img.name}")
            continue
        # hardlink (default), symlink fallback
        try:
            dest.hardlink_to(img)
            print(f"  hardlink {img.name}")
        except OSError as exc:
            dest.symlink_to(img.resolve())
            print(f"  symlink {img.name} (hardlink failed: {exc.strerror})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="Extracted KDZ dir (e.g. V41007h)")
    ap.add_argument("--out", type=Path, required=True, help="Output tree for kdz-tool repack")
    ap.add_argument("--force", action="store_true", help="Overwrite --out")
    ap.add_argument(
        "--clone-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to clone untouched 0.*.img from --base (default: hardlink, "
             "falls back to symlink; avoid copy — system.img is ~24 GiB sparse)",
    )
    ap.add_argument("--sw-version", help="Set dz.sw_version (e.g. V41010d)")
    ap.add_argument("--zip", type=Path, help="V41010d stock installer zip")
    ap.add_argument(
        "--import-zip",
        nargs="+",
        default=[],
        help="Partition names to import from --zip (e.g. system boot laf)",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PART=PATH",
        help="Import raw image for PART from PATH (repeatable)",
    )
    ap.add_argument("--primary-gpt", type=Path, help="Raw/LAF PrimaryGPT.bin")
    ap.add_argument("--backup-gpt", type=Path, help="Raw/LAF BackupGPT.bin (optional)")
    ap.add_argument(
        "--no-trim-zeros",
        action="store_true",
        help="Store full chunk extents instead of trimming trailing zeros",
    )
    ap.add_argument(
        "--dense",
        action="store_true",
        help="Force data_size = sector_count*512 for every imported chunk",
    )
    args = ap.parse_args()

    if not (args.base / "metadata.json").is_file():
        raise SystemExit(f"missing {args.base / 'metadata.json'}")

    copy_base_tree(
        args.base, args.out, force=args.force, clone_mode=args.clone_mode
    )
    meta = load_meta(args.out / "metadata.json")
    trim = not args.no_trim_zeros

    imports: Dict[str, Path] = {}
    if args.zip and args.import_zip:
        imports.update(
            extract_from_zip(args.zip, args.import_zip, args.out / "_raw_imports")
        )
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects PART=PATH, got {item!r}")
        part, path_s = item.split("=", 1)
        imports[part] = Path(path_s)

    for part, raw_path in imports.items():
        if part in ("PrimaryGPT", "BackupGPT"):
            continue
        if not raw_path.is_file():
            raise SystemExit(f"missing raw image: {raw_path}")
        chunks = part_chunks(meta, part)
        out_img = args.out / f"0.{part}.img"
        print(f"Importing {part} from {raw_path} -> {out_img.name}")
        # Replace hardlinked base image
        if out_img.exists():
            out_img.unlink()
        updated = write_sparse_from_raw(
            raw_path,
            out_img,
            chunks,
            trim_zeros=trim,
            dense=args.dense,
        )
        meta["dz"]["parts"]["0"][part] = updated
        total = sum(c["data_size"] for c in updated)
        print(f"  chunks={len(updated)} payload={total / 1024 / 1024:.1f} MiB")

    primary_raw = None
    if args.primary_gpt:
        primary_raw = args.primary_gpt.read_bytes()
        chunks = part_chunks(meta, "PrimaryGPT")
        out_img = args.out / "0.PrimaryGPT.img"
        print(f"Importing PrimaryGPT from {args.primary_gpt}")
        if out_img.exists():
            out_img.unlink()
        meta["dz"]["parts"]["0"]["PrimaryGPT"] = import_primary_gpt(
            args.primary_gpt, out_img, chunks
        )

        # Always refresh BackupGPT to match the new primary
        bchunks = part_chunks(meta, "BackupGPT")
        out_b = args.out / "0.BackupGPT.img"
        print("Synthesizing BackupGPT from PrimaryGPT")
        if out_b.exists():
            out_b.unlink()
        meta["dz"]["parts"]["0"]["BackupGPT"] = import_backup_gpt(
            out_b,
            bchunks,
            primary_raw=primary_raw,
            backup_raw_path=args.backup_gpt,
        )
    elif args.backup_gpt:
        bchunks = part_chunks(meta, "BackupGPT")
        out_b = args.out / "0.BackupGPT.img"
        print(f"Importing BackupGPT from {args.backup_gpt}")
        if out_b.exists():
            out_b.unlink()
        meta["dz"]["parts"]["0"]["BackupGPT"] = import_backup_gpt(
            out_b, bchunks, backup_raw_path=args.backup_gpt
        )

    if args.sw_version:
        meta["dz"]["sw_version"] = args.sw_version
        # Keep KDZ record name readable if it looks like V410*.dz
        for rec in meta.get("kdz", {}).get("records", []):
            name = rec.get("name", "")
            if name.endswith(".dz"):
                rec["name"] = f"{args.sw_version}_0.dz"
        print(f"sw_version -> {args.sw_version}")

    save_meta(args.out / "metadata.json", meta)
    print(
        f"\nReady for repack:\n"
        f"  ./build/kdz-tool repack {args.out} {args.sw_version or 'custom'}.kdz\n"
    )


if __name__ == "__main__":
    main()
