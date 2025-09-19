from fastapi import APIRouter
from api_spec import analyze_brainwave_chunk_api_spec

router = APIRouter(prefix='/api/sleep/brainwave', tags=['brainwave'])

@router.post("/analyze", **analyze_brainwave_chunk_api_spec)
async def analyze_brainwave_chunk(payload: any):
    pass