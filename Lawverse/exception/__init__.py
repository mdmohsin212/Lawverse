import os
import sys

def error_message_deatils(error, error_detail : sys):
    _, _, exc_tab = error_detail.exc_info()
    file_name = exc_tab.tb_frame.f_code.co_filename
    
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tab.tb_lineno, str(error)
    )
    
    return error_message

class ExceptionHandle(Exception):
    def __init__(self, error_message, error_deatil):
        super().__init__(error_message)
        
        self.error_message = error_message_deatils(
            error_message, error_deatil
        )
        
    def __str__(self):
        return self.error_message