import logging

log = logging.Logger(format = '%(filename)s - %(message)s', level=logging.INFO)


def run(runstr: str, *args):
    """run an ML app 
    ex. 
    - run('p2ch11.prepcache.LunaPrepCacheApp')
    - run('training.TrainingApp', '--epochs=1')
    - run('training.TrainingApp', f'--epochs={experiment_epochs}')
    """
    log.info("Running: {}({!r}).main()".format(app, args))

    return 