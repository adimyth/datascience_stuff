import pandas as pd
import multiprocessing as mp
import urllib.request
import rich.traceback
rich.traceback.install()

'''
Given a dataframe of urls & local path storage, downloads the images parallely
'''


def download_images_parallely(data, urls_col, paths_col):
    pool = mp.Pool(processes=mp.cpu_count())
    paths = data[paths_col].tolist()
    urls = data[urls_col].tolist()
    results = [pool.map(download_image, [pair for pair in zip(urls, paths)])]
    pool.close()
    return results


def download_image(pair):
    url = pair[0]
    filename = pair[1]
    try:
        urllib.request.urlretrieve(url, filename)
        return 0
    except:
        return Path(filename).name
