from threading import Thread
import Queue
import time
from tf_sample.crawler import TaskQueue, CrawlPoemUrlWorker


class ThreadWorker(object):
    def __init__(self):
        pass

    def work(self, *args, **kwargs):
        print 'working_'+str(args)+'_'+str(kwargs)

    def close(self):
        print 'close'


def thread_run():
 
    q = TaskQueue(num_workers=2)
    worker = CrawlPoemUrlWorker()
    # worker = ThreadWorker()

    url_list = [
        'http://shici.xpcha.com/poem_3f5te5zq4qf.html',
        'http://shici.xpcha.com/poem_21fte9zq4wu.html',
        'http://shici.xpcha.com/poem_a9dt8czq44u.html',
        'http://shici.xpcha.com/poem_e5ct25zq44y.html',
        'http://shici.xpcha.com/poem_023t84zq449.html',
        'http://shici.xpcha.com/poem_aa0t6folwq9.html',
        'http://shici.xpcha.com/poem_bfdt4dolwxu.html',
        'http://shici.xpcha.com/poem_318tc3ol7be.html',
        'http://shici.xpcha.com/poem_5e8t15ol7l9.html',
        'http://shici.xpcha.com/poem_2d3t7aol7xp.html',
        'http://shici.xpcha.com/poem_fe4tdeomwez.html',
        'http://shici.xpcha.com/poem_f4ct9fomw2u.html',
        'http://shici.xpcha.com/poem_510tcdom7wa.html',
        'http://shici.xpcha.com/poem_e3et46om7xz.html',
        'http://shici.xpcha.com/poem_ae0t4eom74a.html',
        'http://shici.xpcha.com/poem_0aeta2om74j.html',
        'http://shici.xpcha.com/poem_d73t7bom7ez.html',
        'http://shici.xpcha.com/poem_42bte8om7u1.html'
    ]
    for item in url_list:
        q.add_task(worker, item, '456')

    q.join()
    # block until all tasks are done
    print "All done!"
    worker.close()
 
if __name__ == "__main__":
    thread_run()
