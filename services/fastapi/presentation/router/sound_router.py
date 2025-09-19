from fastapi import APIRouter
from api_spec import analyze_sound_chunk_api_spec

router = APIRouter(prefix='/api/sleep/sound', tags=['sound'])

@router.post("/analyze", **analyze_sound_chunk_api_spec)
async def analyze_sound_chunk(payload: any):
    pass