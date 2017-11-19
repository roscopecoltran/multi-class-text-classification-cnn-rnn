import logging
import os
import re

import github
from bs4 import BeautifulSoup
from markdown import markdown
# from main import WORKER_CACHE_DIR

def fetch_readme(repo, cache_prefix_path):

    cache_key = str(repo.full_name)
    cache_dir = cache_prefix_path + os.sep + cache_key

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file = cache_dir + os.sep + 'readme.raw.' + str(repo.id) + '.md'

    logging.info("cache_key: " + cache_key)
    logging.info("cache_dir: " + cache_dir)
    logging.info("cache_file: " + cache_file)

    # check if file is cached
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as file:
            return file.read()

    try:
        readme = repo.get_readme()
        with open(cache_file, 'wt') as outfile:
            content = readme.decoded_content
            content_str = content.decode("utf-8")
            outfile.write(content_str)            
            return content_str
    except github.GithubException:
        # Readme wasn't found
        logging.warning('no readme found for: ' + repo.full_name)
        return ''

    # return readme.decoded_content
    # return readme.content


def markdown_to_text(markdown_string):
    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text