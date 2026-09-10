class APIError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FileReferenceError(APIError):
    def __init__(self, detail: str):
        super().__init__(404, detail)


class CapabilityUnavailableError(APIError):
    def __init__(self, detail: str):
        super().__init__(503, detail)
