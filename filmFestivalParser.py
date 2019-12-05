from lxml import html
import requests
import sys

if __name__ == '__main__':
    if len(sys.argv)>1:
        movieUrl = sys.argv[1]
        page = requests.get(movieUrl)
        movieTree = html.fromstring(page.content)

        # Get movie information
        url = movieUrl
        title = movieTree.xpath("//h1[contains(@class, 'article-title')]/text()")[0].strip()
        origTitle = movieTree.xpath("//header[contains(@class, 'article-header clearfix')]/h3/text()")[0].strip()
        info = movieTree.xpath("//*[contains(@class, 'movie-info-item')]/span/text()")
        plotElem = movieTree.xpath("//section[contains(@itemprop, 'description')]/p/text()")
        if not plotElem:
            plot = movieTree.xpath("//section[contains(@itemprop, 'description')]/text()")[0].strip()
        else:
            plot = plotElem[0].strip()

        infoDict = {info[i - 1]: info[i] for i in range(1, len(info))}
        year = infoDict['Έτος Παραγωγής: ']
        country = infoDict['Χώρα Παραγωγής: ']
        # Output information
        print('{},"{}","{}",{},{},"{}"'.format(url, title, origTitle, year, country, plot))
    else:
        print('Please provide movie url')