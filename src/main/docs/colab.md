

key = '''-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABFwAAAAdzc2gtcn
NhAAAAAwEAAQAAAQEAvXos4FjDNdpJX4yr6UhywPXgxuoAX/Wgja8VlcLrc6OjUeUWYEXB
qojd50Hg6Q3tdPw5B/58Wj9R3VDefXlN8wKg3h5Jxa4wQ8CF7+Sag1pQmgZcW1B2SNBLii
BDLiZFRlh6kpJ82YzEkiyizXMSNPML2f+GMxgHjh289tEWrNf+q2T7zMQ0EVMabDrAWCLh
gn0f80b4nUDIERmHfLHnzpAem0BheN2BU5PEdQvcSHtrIUIlvpHyh6Vkl5Vq3y2P4umNMG
/JA0EDUFi70pdUhh/Y4aKWMKU585CDkteevBvHLAciHWxU3umKQdNdgciy+JDQPJk744+q
qbEQ7EYxaQAAA8jO2gdcztoHXAAAAAdzc2gtcnNhAAABAQC9eizgWMM12klfjKvpSHLA9e
DG6gBf9aCNrxWVwutzo6NR5RZgRcGqiN3nQeDpDe10/DkH/nxaP1HdUN59eU3zAqDeHknF
rjBDwIXv5JqDWlCaBlxbUHZI0EuKIEMuJkVGWHqSknzZjMSSLKLNcxI08wvZ/4YzGAeOHb
z20Ras1/6rZPvMxDQRUxpsOsBYIuGCfR/zRvidQMgRGYd8sefOkB6bQGF43YFTk8R1C9xI
e2shQiW+kfKHpWSXlWrfLY/i6Y0wb8kDQQNQWLvSl1SGH9jhopYwpTnzkIOS1568G8csBy
IdbFTe6YpB012ByLL4kNA8mTvjj6qpsRDsRjFpAAAAAwEAAQAAAQEAki6qZMPWh7vLk/4x
u597eUe5jX2HoIEex3DnFE333ZXIMSyvYMMwsWM64GpBHUzzgKf/UB2UzwO/IyJ7JQ7rhq
rmdbekbvD+p6bnLreORfzt5oc1xfWD7JVXUk+lxPsdwzIMDv0ZebZTCfuJ9zvqNhO1dxDe
9ph5a7mhykJyXhJKv5s3b71QQn3ISD5QpVgdI/3rTAbsQUjhsocany4RtBVx7CT6IaNAAl
oUa8f5Xmh76+wYTQpfzlPcp5tvn1iAawPBSixAX0B6CQjT3AdDse7s7m6JWIEsORMIaz63
YH3+1ydHecHC+2N9Q/5h7msc6hvDKoz5Qzz1slqWpjH6AQAAAIEAgleJQuVQ5bFMRPPbXp
ZZUPx9sQqTb37Y678v47kshpPQvr7GC4guCuoFiJASwIjWsXrf+x9QpRE3tddAmP+dET9q
ANtVUnNXerve3V7A1QqEnQjqohpDcmJnRsz0K1M6bNnZb2Lu1y/voiRZh94oJ5synz4QDM
8c41gJ3t9q6E0AAACBAN0R9OvMUuLJp6p1rbz3BLFv+QKr+Cf7X61IpsBc2ZVSvjGUnOCZ
7PzUGYqlzASfhpcKUqjlFaj3ulWdbYaSv6V6U4XfK/WTYwrcrLavXKtZd4YHIw4LOSDzd4
BgmvFHFnga1drp/7tqeGVIUQp/zjK0hGCgc5Mssl+93lrgErfJAAAAgQDbalIfzcml75JZ
075+zyKR6TS9JA65lYedVS2PPybTdcdszNgRJ9x9YZrRbfmsGvfA1nEXcLJEB5yBNpvR+f
Dbx1X6nATMPWI+CAQiMgcmfXDop0MruCC0XXmZPAX2cMVa8xEF5uomdMN8uxKkmbbM/u8Y
DrGak7xOlIixm8G8oQAAABAyODc2NjU1MzBAcXEuY29tAQ==
-----END OPENSSH PRIVATE KEY-----
'''
with open(r'/root/.ssh/id_rsa', 'w', encoding='utf8') as fh:
    fh.write(key)

pub_key='ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC9eizgWMM12klfjKvpSHLA9eDG6gBf9aCNrxWVwutzo6NR5RZgRcGqiN3nQeDpDe10/DkH/nxaP1HdUN59eU3zAqDeHknFrjBDwIXv5JqDWlCaBlxbUHZI0EuKIEMuJkVGWHqSknzZjMSSLKLNcxI08wvZ/4YzGAeOHbz20Ras1/6rZPvMxDQRUxpsOsBYIuGCfR/zRvidQMgRGYd8sefOkB6bQGF43YFTk8R1C9xIe2shQiW+kfKHpWSXlWrfLY/i6Y0wb8kDQQNQWLvSl1SGH9jhopYwpTnzkIOS1568G8csByIdbFTe6YpB012ByLL4kNA8mTvjj6qpsRDsRjFp 287665530@qq.com'
with open(r'/root/.ssh/id_rsa.pub', 'w', encoding='utf8') as fh:
    fh.write(pub_key)
    
!chmod 600 /root/.ssh/id_rsa.pub
!ssh-keyscan github.com >> /root/.ssh/known_hosts    
!ssh-keyscan github.com >> /root/.ssh/known_hosts
!git config --global user.email "287665530@qq.com"

from google.colab import drive
drive.mount("/content/gdrive")

import os
os.chdir('/content/gdrive/My Drive/Colab Notebooks/workshop/keras-models/src/main/py')

cd /xxxx/xxxx
!git clone git@github.com:jinxueyu/keras-models.git