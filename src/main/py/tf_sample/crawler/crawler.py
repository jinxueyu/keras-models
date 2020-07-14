# -*- coding: utf-8 -*-
import Queue
import cookielib
import threading
import urllib2
import time
import os
import sys
import random
import re

from pyquery import PyQuery as pq

from tf_sample.utils import json
reload(sys)
sys.setdefaultencoding('utf8')

__author__ = 'xueyu'

agent_list = [
    'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
    'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)']


class Crawler(object):
    def __init__(self, name):
        self.crawler_name = name

    parser = None

    def crawl_page(self, url):

        cj = cookielib.CookieJar()
        opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))

        # 简单的迷惑下对方网站
        random_int = random.randint(0, len(agent_list) - 1)
        # time.sleep(random_int * 0.1)
        ua = agent_list[random_int]

        opener.addheaders = [('User-agent', ua)]
        urllib2.install_opener(opener)
        req = urllib2.Request(url)
        req.add_header("Referer", "http://www.baidu.com")
        resp = urllib2.urlopen(req)
        text = None
        try:
            text = resp.read()
            text = text.decode('utf8', 'ignore').encode('utf8', 'ignore')
        except:
            print '[ERROR]Crawler 52:  unicode error', url, encoding, fencoding['confidence']

        return text

    def get_url_list(self, channel):
        pass

    def get_channels(self):
        pass

    def handle_parse_result(self, obj):
        pass

    def time_format(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def parse_page(self, page):
        return self.parser.parse(page)

    def crawl(self):

        self.crawl_page('')

        self.close()

    def close(self):
        pass


class XpchaPageParser(object):

    def multiple_replace(self, text, re_dict):
        rx = re.compile('|'.join(map(re.escape, re_dict)))

        def one_xlat(match):
           return re_dict[match.group(0)]

        return rx.sub(one_xlat, text)

    def parse_list_page(self, text):
        url_list = []
        doc = pq(text)
        hrefs = doc('.bluegpanel')('a')
        for i in range(0, hrefs.length):
            url = hrefs.eq(i).attr('href')
            if url is not None and url.startswith('http://www') and url.find('doctor') > 0:
                url_list.append(url)
        return url_list

    def parse_list_page_urls(self, text):
        page_list = set([])
        # doc = pq(text)
        # arr = doc('.plist')('dt')('a')
        # for h in arr:
        #     page_url = h.get('href')
        #     page_list.append('http://shici.xpcha.com'+page_url[1:])

        url_list = re.findall(r'href="([^"]+)', text)
        for k in url_list:
            if k.startswith('./poem_'):
                page_list.add(k)
        return page_list

    def parse_poet(self, text):
        doc = pq(text)
        poet_box = doc('.main')('.mcon')('.box-bd').eq(1)

        name = poet_box('h2').text()
        tags = poet_box('p').eq(0).text()

        total = int(tags.encode('utf8').split('：')[2])
        page_num = total / 10

        return name, page_num

    def parse(self, text):

        json_obj = {}
        doc = pq(text)

        title_box = doc('.xlarge')
        title = title_box.text()
        tags = title_box.next().text()
        content_box = title_box.next().next().next().next()
        content_box('strong').remove()
        content = content_box.text().encode('utf8')

        content = re.sub('[0-9\[\]\n]+', '', content)

        json_obj['title'] = title.encode('utf8')
        json_obj['content'] = content
        json_obj['tags'] = {}
        tags = tags.encode('utf8')
        tags = re.split('：| ', tags)

        i = 0
        l = len(tags)
        # print json_obj
        # print tags, text
        while l > 0 and i < l:
            k = tags[i]
            i += 1
            v = tags[i]
            i += 1

            json_obj['tags'][k] = v

        return json_obj

    def parse_channel_page(self, text):
        doc = pq(text)

        hrefs = doc('.black_link')('a')
        for href in hrefs:
            href_pq = pq(href)
            print href_pq.text(), '\t', href_pq.attr('href')


class XpchaCrawler(Crawler):
    def __init__(self, file_path, start=0, end=-1):
        Crawler.__init__(self, 'xpcha')
        self.parser = XpchaPageParser()
        channel_file_path = os.path.split(os.path.realpath(__file__))[0]+'/xpcha_channel.txt'
        self.__channel_list = self.read_channel_list(channel_file_path)

        self.__channel_split_start = start
        self.__channel_split_end = end

        if file_path is not None:
            self.__writer = open(file_path, 'w')
        else:
            pass

    __writer = None
    __dao = None

    def read_channel_list(self, file_path):
        channel_list = []
        file_reader = open(file_path)
        while True:
            line = file_reader.readline()
            if not line:
                break
            channel = line.split('    ')
            start = int(channel[1])
            end = int(channel[2])
            channel_list.append((channel[0], start, end))
        file_reader.close()
        return channel_list

    def crawl(self):
        channel_url_list = []
        for channel in self.__channel_list:
            channel_url, start, end = channel
            start += 1
            end += 1
            channel_urls = self.parse_channel_url_list(channel_url, start, end)
            print 'crawl', len(channel_urls),  channel_url, start, end
            channel_url_list.extend(channel_urls)

            for channel_url in channel_urls:
                print '185', channel_url
                page_url_list = self.get_url_list(channel_url)
                for page_url in page_url_list:
                    print '190', page_url
                    page = self.crawl_page(page_url)
                    obj = self.parser.parse(page)
                    line = json.write(obj)
                    self.__writer.write(line+'\n')

    def parse_channel_url_list(self, channel_url, start, end):
        channel_url_list = [channel_url]

        j = channel_url.find('.html')
        channel_url = channel_url[:j]

        for i in range(start, end):
            channel_url_list.append(channel_url+'_'+str(i)+'.html')

        return channel_url_list

    def get_url_list(self, channel_url):
        url_list = []
        page = None
        try:
            page = self.crawl_page(channel_url)
        except Exception, e:
            print 'crawl channel [%s] error' % channel_url, e
            return url_list

        if page is None:
            return url_list

        url_list = self.parser.parse_list_page_urls(page)

        return url_list

    def get_channels(self):
        return self.__channel_list[self.__channel_split_start:self.__channel_split_end]

    def handle_parse_result(self, doctor):

        url = doctor['url']

        doctor_json = json.write(doctor)

        if self.__writer is not None:
            self.__writer.write(doctor_json.encode('utf-8'))
            self.__writer.write('\n')

    def get_poet_list(self, file_path):
        channel_list = self.read_channel_list(file_path)
        channel_url_list = []
        for channel in channel_list:
            url_list = self.parse_channel_url_list(channel[0], channel[1], channel[2])
            channel_url_list.extend(url_list)

        poet_url_list = []
        poet_writer = open('poet_urls.txt', 'w')
        try:
            for channel_url in channel_url_list:
                print channel_url
                url_list = self.get_url_list(channel_url)
                poet_url_list.extend(url_list)
                for u in url_list:
                    poet_writer.write(u+'\n')
        except Exception, e:
            pass
        finally:
            poet_writer.close()
        return poet_url_list

    def close(self):
        if self.__writer is not None:
            self.__writer.close()
        # self.__dao.close()


def crawl_poet(crawler, parser, start, end):
    count = 0
    url_list = []
    reader = open(os.path.split(os.path.realpath(__file__))[0] + '/poet_urls.txt', 'r')
    idx = 0
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.rstrip()
        if start <= idx < end:
            url_list.append(line)
        idx += 1
    poet_writer = open(os.path.split(os.path.realpath(__file__))[0] + '/poet_00' + str(start) + '.txt', 'w+')
    for u in url_list:
        page = crawler.crawl_page(u)
        try:
            name, page_num = parser.parse_poet(page)
        except Exception, e:
            print 'Error Url', u, e
        u = u.replace('poet', 'poemlista')
        poet_writer.write(u + '\t' + name.encode('utf8') + '\t' + str(page_num) + '\n')
        print count
        # u+'\t'+name.encode('utf8')+'\t'+str(page_num)
        count += 1
    poet_writer.close()


def get_filename(filepath, filetype):
    filename = []
    for root, dirs, files in os.walk(filepath):
        for i in files:
            if filetype in i:
                filename.append(filepath+i)
    return filename


def extract_poet_page_urls():
    file_list = get_filename(os.path.split(os.path.realpath(__file__))[0] + '/', '0.txt')
    url_set = set([])
    poet_page_url = []
    for file_name in file_list:
        reader = open(file_name, 'r')
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.rstrip()
            arr = line.split('\t')
            url = arr[0]
            count = int(arr[2])
            if url in url_set:
                continue
            else:
                url_set.add(url)
            poet_page_url.append(url)
            if count == 0:
                continue
            for i in range(1, count + 1):
                u = url.split('.htm')[0] + '_p' + str(i) + '.html'
                poet_page_url.append(u)
        reader.close()
    writer = open('poet_page_urls.txt', 'w')
    for u in poet_page_url:
        writer.write(u + '\n')
    writer.close()


def get_poem_urls():
    file_list = get_filename(os.path.split(os.path.realpath(__file__))[0] + '/', 'poem_urls')
    url_set = set([])
    poet_page_urls = []
    print file_list
    for file_name in file_list:
        if 'poem_urls_test' in file_name:
            continue
        reader = open(file_name, 'r')
        while True:
            line = reader.readline()
            if not line:
                break
            url = line.rstrip()
            if not url.startswith('http'):
                # './poem_96eqa2bfdvy.html'
                url = 'http://shici.xpcha.com'+url[1:]
            if url in url_set:
                continue
            else:
                url_set.add(url)
            poet_page_urls.append(url)
        reader.close()
    writer = open('poem_all_urls.txt', 'w')
    for u in poet_page_urls:
        writer.write(u + '\n')
    writer.close()


class TaskQueue(Queue.Queue):

    def __init__(self, num_workers=1):
        Queue.Queue.__init__(self)
        self.num_workers = num_workers
        self.start_workers()
        self.__parser = XpchaPageParser()
        self.__writer = open('poem_list_test.txt', 'w')

    def add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs))

    def start_workers(self):
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()

    def worker(self):
        count = 0
        while True:
            count += 1
            if count % 1000 == 0:
                print 'write:'+str(count)+'poem'
            worker, args, kwargs = self.get()
            json_obj = None
            try:
                json_obj = self.__parser.parse(args[0])
            except Exception, e:
                print 'parse error:'+args[1]
            if json_obj is not None:
                self.__writer.write(json.write(json_obj)+'\n')
            self.task_done()
        self.__writer.close()


task_flag = 0
def thread_crawl_worker(thread_name, task_list):
    writer = open('poem_urls_'+thread_name+'.txt', 'w')
    for url in task_list:
        page = crawler.crawl_page(url)
        poem_urls = parser.parse_list_page_urls(page)
        l = '\n'.join(poem_urls)
        writer.write(l+'\n')
    writer.close()
    task_flag += 1


class CrawlPoemUrlWorker(object):
    def __init__(self):
        self.__parser = XpchaPageParser()
        self.__writer = open('poem_list_test.txt', 'w')

    def work(self, page, args):
        print 'working_'+str(args)
        json_obj = self.__parser.parse(page)
        self.__writer.write(json.write(json_obj)+'\n')

    def close(self):
        self.__writer.close()


class CrawlThread(threading.Thread):
    def __init__(self, thread_name,  task_list, task_queue):
        threading.Thread.__init__(self)
        self.__task_list = task_list
        self.__thread_name = thread_name
        self.__crawler = XpchaCrawler(None)
        self.__worker = None
        self.__task_queue = task_queue

    def run(self):
        print 'thread run:'+str(self.__thread_name)

        task_count = 0
        for url in self.__task_list:
            task_count += 1
            page = None
            while True:
                try:
                    page = crawler.crawl_page(url)
                except Exception, e:
                    print 'http error retry 8 sec'
                    time.sleep(8)
                    continue

                if page is not None:
                    break

            if task_count % 500 == 0:
                print 'thread task count:'+str(self.__thread_name)+'_task count_'+str(task_count)
            # print 'run_'+url
            self.__task_queue.add_task(self.__worker, page, url)
            pass
        # writer.close()
        self.__crawler.close()
        print 'crawler,', self.__thread_name, 'stop'


def crawl_poem():
    count = 0
    idx = 0
    poem_url_set = set([])
    task_queue = TaskQueue(num_workers=1)
    reader = open('poem_all_urls.txt', 'r')
    task_array = {}
    while True:
        line = reader.readline()
        if not line:
            break
        url = line.rstrip()
        idx += 1

        task_id = idx % 20
        if task_id not in task_array:
            task_array[task_id] = []
        task_array[task_id].append(url)

    thread_pool = []
    task_count = 1
    for task_list in task_array.values():
        print '482 debug:'+str(task_count)
        worker = CrawlThread(str(task_count), task_list, task_queue)
        worker.setDaemon(True)
        worker.start()
        thread_pool.append(worker)
        task_count += 1

    for worker in thread_pool:
        worker.join()
    task_queue.join()

    reader.close()


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        print start, end

    crawler = XpchaCrawler(None)
    # crawler.crawl()
    # page = crawler.crawl_page('http://shici.xpcha.com/poemlistd_7_1000.html')
    # print '1.'
    parser = XpchaPageParser()
    # page_list = parser.parse_list_page_urls(page)
    # print '2.'
    # # page_url = page_list[0]
    # page_url = 'http://shici.xpcha.com/poem_d145d6327fk.html'
    # page = crawler.crawl_page('http://shici.xpcha.com/poemlista_299q73d92ny_p1.html')
    # print '3.'
    # parser.parse(page)
    # print filter(str.isalpha, parser.multiple_replace('abc[3]ss\t', {'[': '', ']': ''}))
    # print re.sub('[0-9①-㊿\[\]]+', '', 'a①b⑫c[3]ss')

    # crawl_poet(crawler, parser, start, end)
    # extract_poet_page_urls()

    # url_list = re.findall(r'href="([^"]+)', page)
    # for k in url_list:
    #     if k.startswith("./poem_"):
    #         print k

    crawl_poem()
    # get_poem_urls()
    crawler.close()

