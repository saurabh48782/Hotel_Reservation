import sys


class CustomException(Exception):

    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_msg(
            error_message, error_detail)

    @staticmethod
    def get_detailed_error_msg(error_message, error_detail):
        _, _, exc_traceback = sys.exc_info()
        if exc_traceback:
            file_name = exc_traceback.tb_frame.f_code.co_filename
            line_num = exc_traceback.tb_lineno

            return (
                f"Error in {file_name} at line {line_num}: {error_message}, "
                f"Exception: {error_detail}")

        return f"{error_message} | Original Exception: {error_detail}"

    def __str__(self):
        return self.error_message
