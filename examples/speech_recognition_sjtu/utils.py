from datetime import datetime
import os

def cacheCommands(argv):
    if not os.path.isdir("CMDs"):
        os.makedirs("CMDs")
    with open("CMDs/command.sh", "a") as f_out:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f_out.write('[Submitted Time: {}]:\t'.format(current_time))
        f_out.write('python {}\n'.format(' '.join(argv)))
