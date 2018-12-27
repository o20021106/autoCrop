import multiprocessing

bind = ['0.0.0.0:5000']
workers = max((multiprocessing.cpu_count() * 2) - 1, 2)
worker_class = 'gevent'
accesslog = '-'
errorlog = '-'
