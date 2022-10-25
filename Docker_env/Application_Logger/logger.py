import datetime

class App_Logger:
    def log(self, file_object, log_message):
        self.now = datetime.datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file = open(file_object, 'a+')
        file.write(str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")
        file.close()
        print(log_message)