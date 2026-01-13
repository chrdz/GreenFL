class Logger:

    def log(message:str):
        return ""



class PrintLooger(Logger):

    def log(message:str):
        print(message)

class FileLogger(Logger):

    def log(message:str):
        # ecrire dans un fichier
        return

class MonProgramme():

    def __init__(logger:Logger):
        my_log = logger
        my_log.log("start")