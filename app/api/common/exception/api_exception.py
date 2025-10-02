class ApiException(Exception):
    status_code: int = 500
    message: str = ''
