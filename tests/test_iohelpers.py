from pathlib import Path

from scripts.utils.iohelpers import iter_image_paths, label_from_path


def create_dummy_structure(tmp_path: Path) -> Path:
    known_root = tmp_path / "faces" / "known"
    (known_root / "alice").mkdir(parents=True, exist_ok=True)
    (known_root / "bob").mkdir(parents=True, exist_ok=True)

    for person in ("alice", "bob"):
        for idx in range(2):
            img_path = known_root / person / f"img{idx}.jpg"
            img_path.write_bytes(b"")
    return known_root


def test_iter_image_paths_finds_expected_files(tmp_path: Path) -> None:
    known_root = create_dummy_structure(tmp_path)
    expected = {p for p in known_root.rglob("*.jpg")}
    found = set(iter_image_paths(known_root))
    assert expected == found


def test_label_from_path_uses_parent_directory(tmp_path: Path) -> None:
    known_root = create_dummy_structure(tmp_path)
    alice_image = next(known_root.joinpath("alice").glob("*.jpg"))
    assert label_from_path(alice_image, known_root) == "alice"

    orphan = known_root / "lonely.jpg"
    orphan.write_bytes(b"")
    assert label_from_path(orphan, known_root) == "lonely"
