"""
Dirty script to download all files from https://www.doomworld.com/idgames/levels/doom/
to given target location
"""

import os
import argparse
import pandas as pd
import numpy as np
import bs4
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
from urllib.request import urlretrieve
import zipfile
import time
import subprocess


MAX_TRIES = 5
WEBSITE_URL = "https://www.doomworld.com"
BASE_URL = "https://www.doomworld.com/idgames/levels/doom/"
IFRAME_REVIEW_BASE = "https://www.doomworld.com/idgames/"  # Objects for review task are simply appended to this
SPECIAL_WAD_PREFIXES = {"Ports": "Ports", "megawads": "megawads", "deathmatch": "deathmatch",}


def is_bad_url(url):
    return ("idgames://" in url) or (".php?" in url) or (url.endswith(".zip")) or (url.endswith(".txt")) or url in [
        "https://www.doomworld.com/idgames/",
        "https://www.doomworld.com/idgames/?search",
        "?search",
        "https://www.doomworld.com/idgames/?top",
        "?top",
        "https://www.doomworld.com/idgames/?random",
        "?random",
        "https://www.doomworld.com/idgames/?poop",
        "?poop",
        "https://www.doomworld.com/idgames/?textmaker",
        "?textmaker",
        "https://www.doomworld.com/idgames/levels/doom/"
    ]

DISALLOWED_URL_STRINGS = ["Home", "Search", "Top Rated", "Random File", "Info", ".txt Generator", "Parent Directory", "Title", "Author", "Size", "Time", "Rating", "Desc"]
DEBUGGING_LEAF_URL = "https://www.doomworld.com/idgames/levels/doom/megawads/bgcomp"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-od", "--output_dir", default="dataset/", type=str, help="Location to make dataset")  # noqa
    parser.add_argument("-ocsv", "--output_pkl", type=str, default="all_wads.pkl", help="Either absolute path, or path relative to output_dir where pickle file containing information about WADs is dumped")  # noqa
    parser.add_argument("-l", "--limit", type=int, default=-1, help="Number of WAD files to download. If < 0, then unlimited.")
    parser.add_argument("-m", "--mirrors", nargs="+", type=str, default=["New York"], help="Which mirror(s) to download from")
    parser.add_argument("--reconstruct", action="store_true", help="Run this after manually unzipping the few weird downloads")
    args = parser.parse_args()
    # convert args.output_pkl to absolute path
    if not(os.path.isabs(args.output_pkl)):
        args.output_pkl = os.path.join(args.output_dir, args.output_pkl)
    else:
        os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "zips"), exist_ok=True)
    return args


def star_img2score(star_img):
    if str(star_img['src']) == 'images/star.gif':
        return 1
    elif str(star_img['src']) == 'images/emptystar.gif':
        return 0
    elif str(star_img['src']) == 'images/emptyhalfstar.gif':
        return 0
    elif str(star_img['src']) == 'images/halfstar.gif':
        return 0.5
    elif str(star_img['src']) == 'images/qstarfarleft.gif':
        return 0.25
    elif str(star_img['src']) == 'images/qstarmidright.gif':
        return 0.75
    elif str(star_img['src']) == 'images/emptyqstarmidleft.gif':
        return 0
    elif str(star_img['src']) == 'images/emptyqstarfarright.gif':
        return 0
    else:
        print("\n\nWarning: star_img={} doesnt follow known template\n\n".format(star_img))
        return 0.0


def parse_ratings(rc):
    try:
        nvotes = int(str(rc.contents[2]).lstrip('\r\n ').rstrip('\r\n ').split(' ')[0][1:])
        star_list = rc.contents[1].contents
        avg_rating = 0.0
        for star_img in star_list:
            avg_rating += star_img2score(star_img)
        return avg_rating, nvotes
    except:
        return 0.0, 0


def get_all_reviews_from_trs(trs):
    all_reviews = []
    for tr in trs:
        if 'class' in tr.attrs:  # seperator
            continue
        else:
            if len(tr.contents) == 3:
                meta, txt = tr.contents[0], tr.contents[1]
                author = meta.contents[0]
                score = 0.0
                for potential_star in meta.contents:
                    if potential_star.name == 'img':
                        score += star_img2score(potential_star)
                txt = str(txt.string)
                all_reviews.append({
                    'review': str(txt),
                    'score': float(str(score)),
                    'author': str(author),
                })
    return all_reviews


def get_all_reviews_from_query(url):
    # Takes in a URL that is IFRAME_REVIEW_BASE + "iframe_review.php?id={}".format(x)
    # Performs the query
    # Gets all reviews
    review_soup = BeautifulSoup(requests.get(url).content)
    trs = review_soup.find_all("tr")
    return get_all_reviews_from_trs(trs)


def get_metadata(soup):
    tr_tags = soup.find_all("tr")
    # Parse the big table above
    for tr_tag in tr_tags:
        cnt = tr_tag.contents
        if isinstance(cnt, list) and len(cnt) == 2:
            if isinstance(cnt[0], bs4.element.NavigableString):
                continue
            if (cnt[0].contents is not None) and (len(cnt[0].contents) > 0):
                if isinstance(cnt[0].contents[0], bs4.element.NavigableString):
                    if str(cnt[0].contents[0]) == 'Description:':
                        desc = cnt[1].contents[0]
                    elif str(cnt[0].contents[0]) == 'Title:':
                        title = cnt[1].contents[0]
                    elif str(cnt[0].contents[0]) == 'Rating:':
                        rating_content = cnt[1]
                        avg_rating, nvotes = parse_ratings(rating_content)
    objects = soup.find_all("object")
    relevant_tables = []
    for table in soup.find_all("table"):
        if ('class' in table.attrs) and ('review' in table['class']):
            relevant_tables.append(table)
    if len(objects) == 1:
        review_obj = objects[0]
        all_reviews = get_all_reviews_from_query(IFRAME_REVIEW_BASE + review_obj['data'])
    elif len(relevant_tables) == 1:
        trs = relevant_tables[0].find_all("tr")
        all_reviews = get_all_reviews_from_trs(trs)
    else:
        print("No reviews found\n\n")
        all_reviews = []
    return {
        'title': str(title).lstrip('\r\n ').rstrip('\r\n '),
        'description': str(desc).lstrip('\r\n ').rstrip('\r\n '),
        'average_rating': float(str(avg_rating)),
        'reviews': all_reviews,
        'votes': int(str(nvotes)),
    }


def get_urls(args):
    # Recursively go through URLS.
    wad_file_dict = {}
    visited = defaultdict(lambda: False)
    current_urls = [BASE_URL]

    # BFS through website.
    while (len(current_urls) > 0) and ((args.limit < 0) or (len(wad_file_dict) < args.limit)):
        next_urls = []
        for current_url in current_urls:
            print("Going through {}".format(current_url))
            if not(((args.limit < 0) or (len(wad_file_dict) < args.limit))):
                break
            current_page = requests.get(current_url, allow_redirects=True)
            try:
                current_soup = BeautifulSoup(current_page.text)
            except:
                print("Page at {} couldn't be souped. Skipping.".format(current_url))
                continue
            anchors = current_soup.find_all('a')
            # Case 1: This is 1 hop from a .zip file
            # Case 2: Need to go deeper
            case1 = False
            candidate_urls = []
            for link_tag in anchors:
                if str(link_tag.string) in args.mirrors:
                    # Falls into Case 1
                    case1 = True
                    # Set metadata
                    # if (str(link_tag['href'])) == 'http://youfailit.net/pub/idgames/levels/doom/0-9/02what.zip':
                    #     import pdb; pdb.set_trace()
                    print('Found download URL: {}'.format(link_tag['href']))

                    wad_file_dict[link_tag['href']] = get_metadata(current_soup)
                elif str(link_tag.string) in DISALLOWED_URL_STRINGS:
                    pass
                elif (link_tag['href'] not in current_url) and not(is_bad_url(link_tag['href'])):
                    # Case 2
                    suffix = str(link_tag['href'])
                    if suffix[0] != "/":
                        suffix = "/idgames/" + suffix
                    new_url = WEBSITE_URL + suffix
                    if not(visited[new_url]) and not(is_bad_url(new_url)):
                        # print("Adding {}".format(new_url))
                        visited[new_url] = True
                        candidate_urls.append(new_url)
            if case1:
                continue
            next_urls.extend(candidate_urls)
        current_urls = next_urls
    return wad_file_dict


def make_targets(args, url_dicts):
    # Takes in url dict and adds a new key to each dict to indicate where to save the file.
    taken_zip_targets = {}
    taken_dir_targets = {}
    for url in url_dicts.keys():
        parts = url.split('/')
        name = parts[-1][:-4]
        special_type = "none"
        for part in parts:
            if part in SPECIAL_WAD_PREFIXES:
                special_type = SPECIAL_WAD_PREFIXES[part]
                break

        url_dicts[url]['name'] = name
        url_dicts[url]['special_type'] = special_type

        uniqueifier = 0
        while (name + "_" + str(uniqueifier)) in taken_zip_targets:
            uniqueifier += 1
        while (name + "_" + str(uniqueifier)) in taken_dir_targets:
            uniqueifier += 1
        taken_zip_targets[name + "_" + str(uniqueifier)] = taken_dir_targets[name + "_" + str(uniqueifier)] = True
        zip_location = os.path.join(args.output_dir, "zips", name + "_" + str(uniqueifier) + ".zip")  # zip file
        dir_location = os.path.join(args.output_dir, name + "_" + str(uniqueifier) + "/")  # Directory
        url_dicts[url]['zip_location'] = zip_location
        url_dicts[url]['dir_location'] = dir_location
    return url_dicts


def perform_download(args, url_dicts):
    for url in url_dicts.keys():
        success = False
        url_dicts[url]['success'] = False
        for tries in range(MAX_TRIES):
            print('[{}/{}] Downloading {}'.format(tries, MAX_TRIES, url))
            try:
                zip_location, _ = urlretrieve(url, url_dicts[url]['zip_location'])
                success = True
                break
            except:
                import pdb; pdb.set_trace()
                continue

        target_dir = url_dicts[url]['dir_location']
        url_dicts[url]["wad_file"] = []
        if not(success):
            print("[PROBLEM] Could not download {}".format(url))
        else:
            # Unzip it to the dir
            try:
                with zipfile.ZipFile(zip_location, 'r') as f:
                    f.extractall(target_dir)

                for filename in os.listdir(target_dir):
                    if filename.endswith(".wad") or filename.endswith(".WAD"):
                        wad_file = os.path.join(target_dir, filename)
                        url_dicts[url]["wad_file"].append(wad_file)
                url_dicts[url]['success'] = True

            except:
                cmd = ["unzip", zip_location, "-d", target_dir]
                print("Using subprocess to extract zipfile.")
                print("$" + " ".join(cmd))
                subprocess.call(cmd)
                for filename in os.listdir(target_dir):
                    if filename.endswith(".wad") or filename.endswith(".WAD"):
                        wad_file = os.path.join(target_dir, filename)
                        url_dicts[url]["wad_file"].append(wad_file)
                url_dicts[url]['success'] = True
    return url_dicts


def reconstruct(args):
    df = pd.read_pickle(args.output_pkl)
    for i in range(df.shape[0]):
        url = df.index[i]
        new_wad_file = []
        target_dir = df.iloc[i].dir_location
        for filename in os.listdir(df.iloc[i].dir_location):
            if filename.endswith(".wad") or filename.endswith(".WAD"):
                wad_file = os.path.join(target_dir, filename)
                new_wad_file.append(wad_file)
        df.set_value(url, 'wad_file', new_wad_file)
    return df


if __name__ == '__main__':
    # Get arguments
    args = get_args()
    if args.reconstruct:
        print("Reconstructing from {}".format(args.output_pkl))
        df = reconstruct(args)
        start = time.time()
        print("Dumping dataframe")
        df.to_pickle(path=args.output_pkl)
        print("Dumped dataframe in {}s".format(time.time() - start))
    else:
        # Generate list of files to download, mapped to corresponding Metadata struct
        start = time.time()
        url_dicts = get_urls(args)
        print("Parsed website in {}s".format(time.time() - start))
        # Generate corresponding dump locations
        url_dicts = make_targets(args, url_dicts)
        # Download and unzip, keep track of the following:
        # online file location, zip location, WAD file location, metadata
        # Dump the dataframe
        print('Starting downloads')
        start = time.time()
        url_dicts = perform_download(args, url_dicts)
        print('Finished downloads in {}s'.format(time.time() - start))
        # Create a pandas dataframe with all the above information
        start = time.time()
        print("Making and dumping dataframe")
        df = pd.DataFrame.from_dict(url_dicts, orient='index')
        df.to_pickle(path=args.output_pkl)
        print("Made and dumped dataframe in {}s".format(time.time() - start))
