
from datetime import datetime, timezone
import re

def get_date(arg):

    try:
        if isinstance(arg, str):
            date = re.sub(r'[^a-zA-Z0-9 \.\-/]', '', arg)
            if bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date)):
                dt = datetime.strptime(date, "%Y-%m-%d")
                dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp()) * 1000

            date = date.lower()
            alpha = re.sub(r'[^a-zA-Z]', '', date)
            if alpha:
                if alpha.startswith('jan') or 'jan' in alpha:
                    date = date.replace(alpha, "01")
                elif alpha.startswith('feb') or 'feb' in alpha:
                    date = date.replace(alpha, "02")
                elif alpha.startswith('mar') or 'mar' in alpha:
                    date = date.replace(alpha, "03")
                elif alpha.startswith('apr') or 'apr' in alpha:
                    date = date.replace(alpha, "04")
                elif alpha.startswith('may') or 'may' in alpha:
                    date = date.replace(alpha, "05")
                elif alpha.startswith('jun') or 'jun' in alpha:
                    date = date.replace(alpha, "06")
                elif alpha.startswith('jul') or 'jul' in alpha:
                    date = date.replace(alpha, "07")
                elif alpha.startswith('aug') or 'aug' in alpha:
                    date = date.replace(alpha, "08")
                elif alpha.startswith('sep') or 'sep' in alpha:
                    date = date.replace(alpha, "09")
                elif alpha.startswith('oct') or 'oct' in alpha:
                    date = date.replace(alpha, "10")
                elif alpha.startswith('nov') or 'nov' in alpha:
                    date = date.replace(alpha, "11")
                elif alpha.startswith('dec') or 'dec' in alpha:
                    date = date.replace(alpha, "12")

            date = date.replace(" ", ".").replace("-", ".").replace("/", ".")
            print('date ', date)
            dt = datetime.strptime(date, "%d.%m.%Y")
            dt = dt.replace(tzinfo=timezone.utc)
            print('date1 ', dt)
            return int(dt.timestamp()) * 1000

    except Exception as e:
        raise e

    return None

print(get_date("04-04-1954"))