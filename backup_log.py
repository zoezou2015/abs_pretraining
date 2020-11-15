
import argparse, os
from shutil import copyfile

def is_backup(traindir):
    import filecmp
    LOG = 'log.txt'
    logfile = os.path.join(traindir, LOG)
    for f in os.listdir(traindir):
        if f.startswith(LOG) and f != LOG:
            bak_log_file = os.path.join(traindir, f)
            if filecmp.cmp(logfile, bak_log_file):
                return True
    return False


def backup_log(traindir):
    logfile = os.path.join(traindir, 'log.txt')
    if os.path.exists(logfile):
        # check if the logfile has been backup
        if is_backup(traindir):
            print('has backup already')
            return

        # backup file
        lid = 1
        while True:
            new_logfile = '{}.{}'.format(logfile, lid)
            if not os.path.exists(new_logfile):
                copyfile(logfile, new_logfile)
                break
            lid += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('traindir')
    args = parser.parse_args()

    backup_log(args.traindir)
