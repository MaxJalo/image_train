import pytest
from io import BytesIO
import zipfile


class TestUploadHandler:
    def create_zip_with_images(self):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr("image1.jpg", b"fake data")
            zf.writestr("image2.png", b"fake data")
        zip_buffer.seek(0)
        return zip_buffer

    def test_zip_validation(self):
        zip_data = self.create_zip_with_images()
        with zipfile.ZipFile(zip_data, 'r') as zf:
            files = zf.namelist()
            assert len(files) == 2

    def test_validate_image_filename(self):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        from pathlib import Path
        assert Path('image.jpg').suffix.lower() in valid_extensions
