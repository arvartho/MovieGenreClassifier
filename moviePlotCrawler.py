from lxml import html
import requests
import csv
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv)>1:
        genreUrls = [sys.argv[1]]
        print('Running crawler with custom argument for genre: {}'.format(genreUrls))
    else:
        mainUrl = 'https://www.filmy.gr/filmy-presentation-movie-database'
        print('Connecting to {}...'.format(mainUrl))
        mainPage = requests.get(mainUrl)
        mainTree = html.fromstring(mainPage.content)
        # Get urls for genre pages
        genreUrls = mainTree.xpath('//blockquote/p/em/strong/a/@href')

    # Prepare lists that will hold the DB attributes
    urls, titles, origTitles, years, countries, genres, plots  = [], [], [], [], [], [], []
    # Dictionary for movie info parsing
    info_dict = {
                'year': 'amy_xronia',
                'country': 'amy_xwra',
                'genre': 'genre-list',
            }

    # Join data parsed in parallel in a single file
    print('Creating csv file to dumb movie info')    
    movieInfoDumb = 'movieDB_filmy.csv'

    csvfile = open(movieInfoDumb, 'a')
    header = ['url', 'title', 'original_title', 'year', 'country', 'genre', 'plot']
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    csvfile.close()
    # Iterate movie www.filmy.gr per genre category
    for genreUrl in genreUrls:
        # Catch "documentary" genre error page
        if genreUrl == 'https://www.filmy.gr/movies-database/back-to-the-top/':
            genreUrl = 'https://www.filmy.gr/genre-list/documentary/'
        print('Getting genre pagination in: {}'.format(genreUrl))

        # Get genre pagination
        genrePage = requests.get(genreUrl)
        genrePageTree = html.fromstring(genrePage.content)

        # Get main url string of the pagination
        # Ex. 'https://www.filmy.gr/genre-list/epic/page'
        pageString = '/'.join(genreUrl.split('/'))

        # Get the last page of the pagination
        if genrePageTree.xpath('//nav/div/a/@href')[-2].split('/'):
            lastPage = int(genrePageTree.xpath('//nav/div/a/@href')[-2].split('/')[-2])

            # Get page url list
            # Ex. ['https://www.filmy.gr/genre-list/epic/page/1',
            #      'https://www.filmy.gr/genre-list/epic/page/2',
            #      'https://www.filmy.gr/genre-list/epic/page/3']
            pageList = ['{}/{}/{}'.format(pageString, 'page', i) for i in range(1, lastPage+1)]
        else:
            pageList = [pageString]

        print('Genre pagination list: {}'.format(pageList))
        # Iterate through the multiple pages of each genre pagination
        for genrePageUrl in pageList:
            print('Parsing genre URL: {}'.format(genrePageUrl))
            # Go to genre page
            movieListPage = requests.get(genrePageUrl)
            movieListPagePageTree = html.fromstring(movieListPage.content)

            # Get movie URL
            movieUrls = movieListPagePageTree.xpath("//*[contains(@class, 'entry-title')]/a/@href")

            # Get movie titles (ex. ['Βασιλιάς Αρθούρος', 'Star Trek', 'Αρμαγεδδών', 'Apocalypto'])
            movieTitle = movieListPagePageTree.xpath("//*[contains(@class, 'entry-title')]/text()")
            movieGenresDiv = movieListPagePageTree.xpath("//div[contains(@class, 'entry-genre')]")

            for moviePageUrl in movieUrls:
                print('Parsing movie URL: {}'.format(moviePageUrl))
                # Go to movie page
                moviePage = requests.get(moviePageUrl)
                moviePageTree = html.fromstring(moviePage.content)

                # Parse movie details:
                # title, original title, year, country, genres, plot
                title = moviePageTree.xpath("//div/h1/a/text()")
                origTitle = moviePageTree.xpath("//li[1]/span/text()")
                
                info_href = moviePageTree.xpath("//li/span/a/@href")
                info = moviePageTree.xpath("//li/span/a/text()")
                
                year_idx = [i for i,label in enumerate(info_href) if info_dict['year'] in label.strip()]
                country_idx = [i for i,label in enumerate(info_href) if info_dict['country'] in label.strip()]
                genre_idx = [i for i,label in enumerate(info_href) if info_dict['genre'] in label.strip()]

                year = ','.join(list(map(lambda x: info[x].strip(), year_idx)))
                country = ','.join(list(map(lambda x: info[x].strip(), country_idx)))
                genre = ','.join(list(map(lambda x: info[x].strip(), genre_idx)))
                # capture the plot by fetching the class 'wpb_wrapper' object with the largest number of characters
                el = moviePageTree.xpath('//div[contains(@class, \'wpb_wrapper\')]/blockquote/p[not(starts-with(text(), ":"))]/text()')
                if len(el)>3:
                    test=1
                    el = moviePageTree.xpath("//div[contains(@class, 'wpb_wrapper')]/blockquote/p/text()")
                else:
                    el = moviePageTree.xpath("//div[contains(@class, 'wpb_wrapper')]//p/text()")
                    test=2
                plot_idx = np.argsort(list(map(len, el)))[-1]
                plot = el[plot_idx].strip()
    
                csvfile = open(movieInfoDumb, 'a')
                writer = csv.writer(csvfile)
                writer.writerow(list(map(lambda x: str(x).strip(), [moviePageUrl, title, origTitle, year, country, genre, plot])))
                csvfile.close()

    print('Finished parsing https://www.filmy.gr')
