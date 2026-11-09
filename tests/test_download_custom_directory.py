import hashlib
from src.download import download_one


def test_existing_download_in_custom_directory(tmp_path):
    directory = tmp_path / "custom-data"
    directory.mkdir()
    content = b"date,OT\n2016-07-01,1.5\n"
    (directory / "ETTh1.csv").write_bytes(content)
    result = download_one("ETTh1", dest_dir=directory)
    assert result["local_path"] == "custom-data/ETTh1.csv"
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["byte_count"] == len(content)
