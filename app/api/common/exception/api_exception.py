class ApiException(Exception):
    status_code: int = 500
    message: str = ''
    # Code key prefix used to build ApiResponse.code like "AUTH401"
    code_key: str = 'API'

    def build_code(self) -> str:
        try:
            return f"{self.code_key}{int(self.status_code)}"
        except Exception:
            return f"API{int(self.status_code)}"
