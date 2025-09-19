from api_exception import ApiException

class InternalApiException(ApiException):
    status_code = 500
    message = '서버 내부 오류입니다.'

class AnalyzeUnavailableException(ApiException):
    status_code = 503
    message = '현재 분석을 진행할 수 없습니다.'

class BadReqeustApiException(ApiException):
    status_code = 400
    message = ''