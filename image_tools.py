# tools/image_tools.py
import os
import io
import requests
from PIL import Image
import rembg
from pydantic import BaseModel
from langchain.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables (later if you want Supabase, keys can live in .env)
load_dotenv()

class BgInput(BaseModel):
    image_url: str

class BackgroundRemovalTool(BaseTool):
    name: str = "background_removal"
    description: str = "Remove background from an image URL and save locally; returns file path."
    args_schema = BgInput

    def _run(self, image_url: str) -> str:
        # 1) download image
        r = requests.get(image_url, timeout=20)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")

        # 2) remove background
        out = rembg.remove(img)

        # 3) save locally in outputs/
        os.makedirs("outputs", exist_ok=True)
        file_path = os.path.join("outputs", f"bg_removed_{abs(hash(image_url))}.png")

        out.save(file_path, format="PNG")

        return file_path

    async def _arun(self, image_url: str) -> str:
        return self._run(image_url)
