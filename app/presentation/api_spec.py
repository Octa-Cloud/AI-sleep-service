analyze_sound_chunk_api_spec = dict(
    status_code=202,
    summary='음성 데이터 청크 분석 요청',
    description='15분 음성 데이터 청크를 서버로 전송합니다. 분석은 비동기로 진행됩니다',
)

analyze_brainwave_chunk_api_spec = dict(
    status_code=202,
    summary='뇌파 청크 분석 요청',
    description='15분 뇌파 데이터 청크를 서버로 전송합니다. 분석은 비동기로 진행됩니다',
)