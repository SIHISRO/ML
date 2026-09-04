import os
import shutil
from typing import List
from fastapi import Request
from starlette.datastructures import UploadFile
from src.core.constants import PUBLIC_TEMP_DIR



os.makedirs(PUBLIC_TEMP_DIR, exist_ok=True)

# takes list of files as input
async def multer_middleware(
    request: Request,
):
    saved_paths: List[str] = []
    try:
        form = await request.form()
        files = [
            value for _, value in form.multi_items()
            if isinstance(value, UploadFile)
        ]
        if not files:
            # logger.info("multer_middleware: no files in request")
            yield []
            return

        thread_id = getattr(request.state, "thread_id", None)
        if not thread_id:
            # logger.warning("multer_middleware: thread_id not found in request state, using fallback")
            thread_id = "default"

        dest_dir = os.path.join(PUBLIC_TEMP_DIR, str(thread_id))
        os.makedirs(dest_dir, exist_ok=True)
        for idx, file in enumerate(files):
            if not file or not file.filename:
                continue
            file_path = os.path.join(dest_dir, f"{idx}_{file.filename}")
            # logger.info("multer_middleware: saving file to %s", file_path)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append(file_path)
            # logger.info("multer_middleware: saved %s", file_path)

        # logger.info("multer_middleware: %d file(s) saved", len(saved_paths))
        yield saved_paths

    except Exception:
        # logger.error("multer_middleware: error saving files: %s", str(e))
        # raise MyException(e, sys)
        raise

    finally:
        for file_path in saved_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # logger.info("multer_middleware: cleaned up %s", file_path)
            except Exception as clean_err:
                # logger.error("multer_middleware: failed to clean up %s: %s", file_path, clean_err)
                pass
        # Remove dir if empty
        if saved_paths:
            dest_dir = os.path.dirname(saved_paths[0])
            if os.path.exists(dest_dir) and not os.listdir(dest_dir):
                os.rmdir(dest_dir)