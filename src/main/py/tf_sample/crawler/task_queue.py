import Queue
import threading
import time

from tf_sample.utils import json
from tf_sample.crawler import XpchaCrawler, XpchaPageParser

__author__ = 'xueyu'


class JobTask(Queue.Queue):

    def __init__(self, next_task, num_workers, produce_func):
        Queue.Queue.__init__(self)
        self.next_task = next_task
        self.num_workers = num_workers
        self.start_workers()
        self.produce_func = produce_func

    def add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs))

    def start_workers(self):
        print '==========='+str(self.num_workers)
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()

    def worker(self):
        while True:
            item, args, kwargs = self.get()
            self.produce(item, args, kwargs)
            self.task_done()

    def produce(self, worker, args, kwargs):
        self.produce_func(worker, args, kwargs)

    def join_all(self):
        self.join()
        if self.next_task is not None:
            self.next_task.join_all()


class CrawlTask(JobTask):

    def __init__(self, next_task, num_workers, produce):
        JobTask.__init__(self, next_task, num_workers, produce)
        self.crawler = XpchaCrawler(None)

        self.task_count = 0
        self.__thread_name = str(time.time())

    def produce(self, url, args, kwargs):
        while True:
            try:
                page = self.crawler.crawl_page(url)
            except Exception, e:
                print 'http error retry 8 sec'
                time.sleep(8)
                continue

            if page is not None:
                break

        if self.task_count % 500 == 0:
            print 'thread task count:'+str(self.__thread_name)+'_task count_'+str(self.task_count)
        # print 'run_'+url
        self.next_task.add_task(page, url)

page_parser = XpchaPageParser()


def parse_poem(item, args, kwargs):
    json_obj = None
    # try:
    json_obj = page_parser.parse(item)
    # except Exception, e:
    #     print 'parse error:'+args[0], e

    print json.write(json_obj)

if __name__ == '__main__':
    job1 = JobTask(None, 1, parse_poem)
    job0 = CrawlTask(job1, 3, None)

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
        job0.add_task(item, '456')

    job0.join_all()
