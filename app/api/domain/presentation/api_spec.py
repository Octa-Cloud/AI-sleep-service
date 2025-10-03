analyze_brainwave_chunk_api_spec = dict(
    status_code=200,
    summary='뇌파 데이터 업로드',
    description='10분 길이의 EDF 데이터를 업로드하면 즉시 수락 응답을 반환합니다. 분석/저장은 비동기로 처리됩니다.',
    method='PATCH',
    path='/api/sleep/data/brainwave/',
    request={'content-type': 'multipart/form-data', 'fields': {'file_instance': 'EDF binary file'}},
    auth={'header': 'Authorization', 'scheme': 'Bearer <JWT>'},
    response={
        '200': {'code': 'COMMON200', 'message': '요청에 성공하였습니다', 'result': {'accepted': True}},
        '400': {'code': 'BRAIN_FORMAT400|BRAIN_CHANNEL400|BRAIN_LENGTH400', 'message': '유효성 검증 실패'},
        '401': {'code': 'AUTH401', 'message': '인증이 필요합니다.'},
    },
)

create_sleep_session_api_spec = dict(
    status_code=200,
    summary='수면 세션 생성',
    description='현재 사용자에 대한 진행 중 수면 세션을 생성합니다. 세션 ID는 응답에 포함되지 않습니다.',
    method='POST',
    path='/api/sleep/session/',
    auth={'header': 'Authorization', 'scheme': 'Bearer <JWT>'},
    response={
        '200': {'code': 'COMMON200', 'message': '요청에 성공하였습니다', 'result': None},
        '401': {'code': 'AUTH401', 'message': '인증이 필요합니다.'},
        '409': {'code': 'SESSION_EXISTS409', 'message': '세션이 이미 존재합니다.'},
    },
)