# core packages
import datetime
import getpass
import logging
import os
from os.path import expanduser
import re
import sys
import shutil
import json
# import yaml
import ruamel.yaml

# natural text processing
import numpy
import github
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
# import pandas as pd
# import spacy

# load .env file variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# readme processors
import readmereader

# github settings
login_or_token          = ""
password                = ""
target_account          = ""

# worker params
get_topics              = False
github_per_page         = 250

# classifier
number_of_topics        = 50

# cache default values
# WORKER_CACHE_DIR        = 'cache'
cache_prefix_path       = ""

CACHE_DIRNAME_ROOT      = cache_prefix_path
CACHE_DIRNAME_CONTENT   = 'content'
CACHE_DIRNAME_CLASSIFY  = 'classify'
CACHE_DIRNAME_REQUEST   = 'requests'
CACHE_DIRNAME_GITHUB    = 'github'

def main():
    g = None
    login_or_token          = ""
    password                = ""
    target_account          = ""

    if os.environ.get('GITHUB_LOGIN_OR_TOKEN') != "":
        login_or_token = os.environ.get('GITHUB_LOGIN_OR_TOKEN')
        g = github.Github(login_or_token)
    elif os.environ.get('GITHUB_API_CLIENT_ID') != "" and os.environ.get('GITHUB_API_CLIENT_SECRET') != "":
        g = github.Github(client_id=os.environ.get('GITHUB_API_CLIENT_ID'), client_secret=os.environ.get('GITHUB_API_CLIENT_SECRET'))
    else:
        login_or_token = input('Your Github Account or Personal Token: ')
        print('Note:')
        print('- User\'s password is not stored in any way...')
        print('- We recommend to use a personal token for authentication.')
        password = getpass.getpass('Your Github Password: ')
        g = github.Github(login_or_token, password)

    if os.environ.get('GITHUB_API_PER_PAGE') != "":
        g.per_page = int(os.environ.get('GITHUB_API_PER_PAGE'))  # maximum allowed value
    else:
        g.per_page = github_per_page  # maximum allowed value is 250

    if os.environ.get('WORKER_USER_AGENT') != "":
        g.user_agent = os.environ.get('WORKER_USER_AGENT')

    if os.environ.get('WORKER_TOPICS_COUNT_MAX') != "":
        number_of_topics = int(os.environ.get('WORKER_TOPICS_COUNT_MAX'))

    if os.environ.get('GITHUB_API_GET_TOPICS') == "True":
        get_topics = True

    if os.environ.get('WORKER_TARGET_ACCOUNT') != "":
        target_account = os.environ.get('WORKER_TARGET_ACCOUNT') 
    else:
        target_account = input('Github username to analyze: ')

    if os.environ.get('WORKER_CACHE_DIR') != "":
        user_home = expanduser("~")
        cache_prefix_path = os.environ.get('WORKER_CACHE_DIR').replace("~", user_home)

    print('   - cache_prefix_path: ' , cache_prefix_path)
    if not os.path.isdir(cache_prefix_path) and cache_prefix_path != "":
        os.makedirs(cache_prefix_path)

    # log info
    # logging.info("\n - GITHUB INFO:")
    # logging.info('   - GITHUB_LOGIN_OR_TOKEN: ' + login_or_token)
    # logging.info('   - GITHUB_API_PER_PAGE: '+ str(g.per_page))

    # logging.info("\n - WORKER INFO:")
    # logging.info('   - WORKER_TARGET_ACCOUNT: ' + str(target_account))

    print("\n - GITHUB INFO:")
    print('   - GITHUB_LOGIN_OR_TOKEN: ' , login_or_token)
    print('   - GITHUB_API_PER_PAGE: ' , str(g.per_page))

    print("\n - WORKER INFO:")
    print('   - WORKER_TARGET_ACCOUNT: ' , str(target_account))

    # sys.exit('Debug - [END]')

    target_user = g.get_user(target_account)
    repos = target_user.get_starred()

    print('   - WORKER_ITEMS_COUNT: ' , repos.totalCount)    

    # setup output directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_directory = 'topics/%s/%s' % (target_account, timestamp)

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # create cache folder
    print('   - TASK_TIMESTAMP: ' , timestamp)
    print('   - TASK_OUTPUT_DIRECTORY: ' , output_directory)
    print()
    print("\n - START:")
    print('   - extracts texts for repos (readmes, etc.)...')

    if not cache_prefix_path.endswith('/'):
        cache_prefix_path += '/'

    # logging.info('   - extracts texts for repos (readmes, etc.)...')
    texts, text_index_to_repo = extract_texts_from_repos(repos, get_topics, cache_prefix_path)

    # Classifying
    vectorizer = TfidfVectorizer(max_df=0.2, min_df=2, max_features=1000, stop_words='english', norm='l2',
                                 sublinear_tf=True)
    vectors = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names()

    decomposition = NMF(n_components=number_of_topics)
    model = decomposition.fit_transform(vectors)

    # generate overview readme
    overview_text = generate_overview_readme(decomposition, feature_names, target_account)

    # README to get displayed by github when opening directory
    overview_filename = output_directory + os.sep + 'README.md'
    logging.info('overview_filename: '+ overview_filename)
    with open(overview_filename, 'w') as overview_file:
        overview_file.write(overview_text)

    # generate topic folders and readmes
    print()
    for topic_idx, topic in enumerate(decomposition.components_):
        top_feature_indices = topic.argsort()[:-11:-1]
        top_feature_names = [feature_names[i] for i in top_feature_indices]

        repo_indices_asc = model[:, topic_idx].argsort()
        repo_indices_desc = numpy.flip(repo_indices_asc, 0)

        print("Topic #%d:" % topic_idx)
        print(", ".join(top_feature_names))

        # output repos
        max_weight = model[repo_indices_desc[0], topic_idx]
        for i in repo_indices_desc:
            weight = model[i, topic_idx]
            if weight > 0.05 * max_weight:
                print(text_index_to_repo[i], weight)

        # create topic directory
        topic_directory_name = "-".join(top_feature_names[0:3])
        topic_path = output_directory + os.sep + topic_directory_name
        os.mkdir(topic_path)

        # generate readme
        topic_readme_text = "# Repositories defined by: %s\n" % ", ".join(top_feature_names[0:3])
        topic_readme_text += '\n'
        topic_readme_text += "also defined by the following keywords: %s\n" % ", ".join(top_feature_names[3:])
        topic_readme_text += '\n'
        for repo in [text_index_to_repo[i] for i in repo_indices_desc if model[i, topic_idx] > 0.1 * max_weight]:
            topic_readme_text += '- [%s](%s)\n' % (repo.full_name, repo.html_url)
            if repo.description:
                topic_readme_text += '  %s\n' % repo.description

        # write readme
        with open(topic_path + os.sep + 'README.md', 'w') as file:
            file.write(topic_readme_text)
        print()


def generate_overview_readme(decomposition, feature_names, username):
    text = '# %s\'s stars by topic\n' % username
    text += '\n'
    text += 'This is a list of topics covered by the starred repositories of %s.' % username
    text += '\n'

    topic_list = []
    for topic_idx, topic in enumerate(decomposition.components_):
        top_feature_indices = topic.argsort()[:-11:-1]
        top_feature_names = [feature_names[i] for i in top_feature_indices]

        topic_name = ", ".join(top_feature_names[0:3])

        topic_directory_name = "-".join(top_feature_names[0:3])
        topic_link = topic_directory_name + os.sep + 'README.md'

        topic_list_item = '- [%s](%s)' % (topic_name, topic_link)
        topic_list.append(topic_list_item)

    topic_list.sort()  # sort alphabetically
    text += '\n'.join(topic_list)

    return text


def extract_texts_from_repos(repos, get_topics, cache_prefix_path):
    readmes = []
    readme_to_repo = {}  # maps readme index to repo

    # print(" #### PaginatedList KEYS: ", repos._PaginatedList__totalCount)
    # print(" #### repos sub-count: ",  repos.length)
    print('   - WORKER_ITEMS_COUNT: ' , repos.totalCount)   
    # dict_keys(['_PaginatedListBase__elements', '_PaginatedList__requester', '_PaginatedList__contentClass', '_PaginatedList__firstUrl', '_PaginatedList__firstParams', '_PaginatedList__nextUrl', '_PaginatedList__nextParams', '_PaginatedList__headers', '_PaginatedList__list_item', '_reversed', '_PaginatedList__totalCount'])
    # logging.info('   - extract_texts_from_repos count: '+ len(repos))

    for repo in repos:
        full_repo_text = get_text_for_repo(repo, get_topics, cache_prefix_path)
        readme_to_repo[len(readmes)] = repo
        readmes.append(full_repo_text)

    return readmes, readme_to_repo


def get_text_for_repo(repo, get_topics, cache_prefix_path):

    repo_login, repo_name = repo.full_name.split('/')  # use full name to infer user login

    cache_prefix = cache_prefix_path + str(repo.full_name) + os.sep + 'repository.api-v3.' + str(repo.id)
    logging.info('   - cache_prefix: ' , cache_prefix)

    readme = readmereader.fetch_readme(repo, cache_prefix_path)
    # readme = readmereader.fetch_readme(user_login, repo_name, repo.id)
    readme_text = readmereader.markdown_to_text(readme)
    repo_name_clean = re.sub(r'[^A-z]+', ' ', repo_name)

    if not os.path.isfile(cache_prefix + '.yaml'):
        with open(cache_prefix + '.yaml', 'w') as outfile:
            # yaml.safe_dump(repo.raw_data, outfile, default_flow_style=False)
            ruamel.yaml.safe_dump(repo.raw_data, outfile, default_flow_style=False)


    if not os.path.isfile(cache_prefix + '.json'):
        with open(cache_prefix + '.json', 'w') as outfile:
            json.dump(repo.raw_data, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    print("---> repo.topics: " , str(repo.topics))
    print(repo.topics)

    repo_topics = ""
    if get_topics:
        if repo.topics is not None:
            repo_topics = str(', '.join(repo.topics))
            logging.info('   - repo_topics: '+ repo_topics)
        else:
            logging.info('   - repo_topics: [NOT_AVAILABLE]')

    texts = [
        str(repo.description),
        str(repo.description),
        str(repo.description),  # use description 3x to increase weight
        str(repo.language),
        str(repo_topics),
        readme_text,
        repo_name_clean
    ]

    return ' '.join(texts)


if __name__ == '__main__':
    main()