import datetime
import sys
from data_upload import upload

if __name__ == "__main__":
    yesterday= datetime.datetime.now() - datetime.timedelta(days=1)
    date = yesterday.strftime("%y%m%d")
    location = sys.argv[2]
    upload(date, "/home/awazon/nsystem/main7.py", location, True, True, True)
