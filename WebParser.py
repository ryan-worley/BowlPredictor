from bs4 import BeautifulSoup
import os
import numpy as np
import collections
import pickle
import math
import urllib.request as request
import requests
import pandas as pd
import collections
import re

def requestBowls(links, start_rows, years):
    """
    :param page:
           start_row:
    :return: bowl_games:
    """
    bowl_games = []
    year = []

    for index, linkyear in enumerate(links):
        print('Getting Links for year {}'.format(years[index]))
        page = requests.get(linkyear)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find(class_='sortable').tbody
        lines = table.find_all('tr')
        for line in lines:
            try:
                gamekey = int(line.find('th').text)
                if gamekey > start_rows[index]:
                    bowl_games.append(line)
                    year.append(years[index])
            except: continue
    return bowl_games, year

def parseLink(bowl_games):
    links = collections.defaultdict(list)
    teams = collections.defaultdict(list)
    team_link = {}
    beginning = 'https://www.sports-reference.com'
    np_links = []
    np_teams = []
    games = []
    score = []
    for i, game in enumerate(bowl_games):
        gamelinks = game.find_all('a')
        links_ = []
        teams_ = []

        for link in gamelinks:
            if 'boxscores' in link['href']:
                gamelinks.remove(link)

        for link in gamelinks:
            links_.append(beginning + link['href'])
            teams_.append(link.text)

        links['Game {}'.format(i + 1)] = links_
        teams['Game {}'.format(i + 1)] = teams_
        team_link['Game {}'.format(i + 1)] = {teams_[0]: links_[0],
                                              teams_[1]: links_[1]}
        np_links.append(links_[0])
        np_links.append(links_[1])
        np_teams.append(teams_[0])
        np_teams.append(teams_[1])
        games.append('G' + str(i+1) + 'T1')
        games.append('G' + str(i+1) + 'T2')

        del links_
        del teams_

    return links, teams, team_link, np_links, np_teams, games

def parseLinkPrevious(bowl_games, year):
    links = collections.defaultdict(list)
    teams = collections.defaultdict(list)
    team_link = {}
    beginning = 'https://www.sports-reference.com'
    np_links = []
    np_teams = []
    games = []
    score_ = []
    game_counter = 1
    for i, game in enumerate(bowl_games):
        gamelinks = game.find_all('a')
        links_ = []
        teams_ = []

        for link in gamelinks:
            if 'boxscores' in link['href']:
                gamelinks.remove(link)

        for link in gamelinks:
            links_.append(beginning + link['href'])
            teams_.append(link.text)

        links['Game {}'.format(i + 1)] = links_
        teams['Game {}'.format(i + 1)] = teams_
        team_link['Game {}'.format(i + 1)] = {teams_[0]: links_[0],
                                              teams_[1]: links_[1]}

        winner = game.find('td', attrs={'data-stat': 'winner_points'}).text
        loser = game.find('td', attrs={'data-stat': 'loser_points'}).text

        score_.append(winner)
        score_.append(loser)
        np_links.append(links_[0])
        np_links.append(links_[1])
        np_teams.append(teams_[0])
        np_teams.append(teams_[1])
        del links_
        del teams_

        games.append('G' + str(game_counter) + 'T1' + year[i])
        games.append('G' + str(game_counter) + 'T2' + year[i])
        game_counter += 1

        if i == len(bowl_games)-1: continue
        elif year[i] != year[i+1]:
            game_counter = 1

    return links, teams, team_link, np_links, np_teams, games, score_

def createCurrentDataFrame(games, nplinks, npteams):
    categories = ['Team', 'Link', 'total_wins', 'win_percentage', 'conf_wins', 'ppg', 'srs', 'sos', 'g', 'pass_att',
                  'pass_cmp_pct',
                  'pass_yds', 'pass_td', 'rush_yds', 'rush_yds_per_att', 'rush_td', 'tot_plays', 'tot_yds',
                  'tot_yds_per_play', 'first_down',
                  'penalty', 'turnovers', 'opp_pass_cmp', 'opp_pass_att', 'opp_pass_cmp_pct', 'opp_pass_yds',
                  'opp_pass_td', 'opp_tot_yds', 'opp_tot_yds_per_play',
                  'opp_first_down', 'opp_penalty', 'opp_turnovers']
    bd = pd.DataFrame(index=categories, columns=games)

    for i, link in enumerate(nplinks):
        game = games[i]
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'lxml')
        print('Mining Link: ', link)

        bd[game]['Link'] = link
        bd[game]['Team'] = npteams[i]

        top = soup.find('div', id='meta')
        chart = top.find('div', attrs={'data-template': 'Partials/Teams/Summary'})
        record = re.split('[, \-!?:]', top.find('strong', text='Record:').parent.text)
        bd[game]['total_wins'] = int(record[2])
        bd[game]['conference'] = (top.find('strong', text='Conference:').parent.text.split()[1:]).join()
        bd[game]['win_percentage'] = float(record[2]) / (float(record[2]) + float(record[3]))
        bd[game]['conf_wins'] = int(re.split('[, \-!?:]', top.find('strong', text='Conference Record:').parent.text)[3])
        bd[game]['ppg'] = float(re.split('[, \-!?:]', top.find('strong', text='Points/G:').parent.text)[2])
        bd[game]['srs'] = float(re.split('[, !?:]', top.find('strong', text='SRS').parent.parent.text)[2])
        bd[game]['sos'] = float(re.split('[, !?:]', top.find('strong', text='SOS').parent.parent.text)[2])

        team_table = soup.find('table', class_='sortable', id='team')
        body = team_table.find('tbody')

        bd.at['g', game] = body.find('td', attrs={'data-stat': 'g'}).text
        bd.at['pass_cmp', game] = body.find('td', attrs={'data-stat': 'pass_cmp'}).text
        bd.at['pass_att', game] = body.find('td', attrs={'data-stat': 'pass_att'}).text
        bd.at['pass_cmp_pct', game] = body.find('td', attrs={'data-stat': 'pass_cmp_pct'}).text
        bd.set_value('pass_yds', game, body.find('td', attrs={'data-stat': 'pass_yds'}).text)
        bd.set_value('pass_td', game, body.find('td', attrs={'data-stat': 'pass_td'}).text)
        bd.set_value('rush_yds', game, body.find('td', attrs={'data-stat': 'rush_yds'}).text)
        bd.set_value('rush_yds_per_att', game, body.find('td', attrs={'data-stat': 'rush_yds_per_att'}).text)
        bd.set_value('rush_td', game, body.find('td', attrs={'data-stat': 'rush_td'}).text)
        bd.set_value('tot_plays', game, body.find('td', attrs={'data-stat': 'tot_plays'}).text)
        bd.set_value('tot_yds', game, body.find('td', attrs={'data-stat': 'tot_yds'}).text)
        bd.set_value('tot_yds_per_play', game, body.find('td', attrs={'data-stat': 'tot_yds_per_play'}).text)
        bd.set_value('first_down', game, body.find('td', attrs={'data-stat': 'first_down'}).text)
        bd.set_value('penalty', game, body.find('td', attrs={'data-stat': 'penalty'}).text)
        bd.set_value('turnovers', game, body.find('td', attrs={'data-stat': 'turnovers'}).text)
        bd.set_value('opp_pass_cmp', game, body.find('td', attrs={'data-stat': 'opp_pass_cmp'}).text)
        bd.set_value('opp_pass_att', game, body.find('td', attrs={'data-stat': 'opp_pass_att'}).text)
        bd.set_value('opp_pass_cmp_pct', game, body.find('td', attrs={'data-stat': 'opp_pass_cmp_pct'}).text)
        bd.set_value('opp_pass_yds', game, body.find('td', attrs={'data-stat': 'opp_pass_yds'}).text)
        bd.set_value('opp_pass_td', game, body.find('td', attrs={'data-stat': 'opp_pass_td'}).text)
        bd.set_value('opp_tot_yds', game, body.find('td', attrs={'data-stat': 'opp_tot_yds'}).text)
        bd.set_value('opp_tot_yds_per_play', game, body.find('td', attrs={'data-stat': 'opp_tot_yds_per_play'}).text)
        bd.set_value('opp_first_down', game, body.find('td', attrs={'data-stat': 'opp_first_down'}).text)
        bd.set_value('opp_penalty', game, body.find('td', attrs={'data-stat': 'opp_penalty'}).text)
        bd.set_value('opp_turnovers', game, body.find('td', attrs={'data-stat': 'opp_turnovers'}).text)

    bdt = bd.T
    bd.to_csv('2018_Data_Transpose.csv')
    bdt.to_csv('2018_Data.csv')


def bowlLinks(stats_links):
    return [link[:-5] + '/gamelog/' for link in stats_links]


def adjustAverage(total, bowl, games):
    '''
    >>> adjustAverage(12, 1, 12)
    13.0
    '''
    return (total*games - bowl) / (games - 1)


def createPreviousDataFrame(games, nplinks, npteams, filename):
    categories = ['Team', 'Link', 'Conference', 'total_wins', 'win_percentage', 'ppg', 'srs', 'sos', 'g', 'pass_att',
                  'pass_cmp_pct', 'pass_cmp',
                  'pass_yds', 'pass_td', 'rush_yds', 'rush_att', 'rush_yds_per_att', 'rush_td', 'tot_plays', 'tot_yds',
                  'tot_yds_per_play', 'first_down',
                  'penalty', 'turnovers', 'opp_pass_cmp', 'opp_pass_att', 'opp_pass_cmp_pct', 'opp_pass_yds',
                  'opp_pass_td', 'opp_tot_yds', 'opp_tot_yds_per_play',
                  'opp_first_down', 'opp_penalty', 'opp_turnovers']

    # bd = pd.DataFrame(index=categories, columns=games)
    bd = pd.DataFrame(columns=games)
    bowl_links = bowlLinks(nplinks)

    for i, link in enumerate(nplinks):
        game = games[i]
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'lxml')
        team_table = soup.find('table', class_='sortable', id='team')
        body = team_table.find('tbody')
        print('Mining Link: ', link)

        page = requests.get(bowl_links[i])
        bowl_of_soup = BeautifulSoup(page.content, "lxml")
        bowl_tables = bowl_of_soup.find_all('table', class_='sortable')
        offensive_table = bowl_of_soup.find_all('table')[0]

        bd.at['Link', game] = link
        bd.at['Team', game] = npteams[i]
        total_games_played = int(body.find('td', attrs={'data-stat': 'g'}).text)
        bd.at['g', game] = total_games_played-1

        top = soup.find('div', id='meta')
        chart = top.find('div', attrs={'data-template': 'Partials/Teams/Summary'})
        record = re.split('[, \-!?:]', top.find('strong', text='Record:').parent.text)
        conference = (top.find('strong', text='Conference:').parent.text.split())
        bd.at['Conference', game] = ' '.join(conference[1:])

        bg_offense = offensive_table.find('th', attrs={'csk': total_games_played}).parent
        result = bg_offense.find('td', attrs={'data-stat': 'game_result'}).text
        info = list(filter(None, re.split('[, \-()!?:]', result)))
        pf = info[1]
        pa = info[2]
        result = info[0]

        if result == 'W' or result == 'w':
            bd.at['total_wins', game] = int(record[2]) - 1
            bd.at['win_percentage', game] = (float(record[2]) - 1) / (float(record[2]) + float(record[3])-1)
        else:
            bd.at['total_wins', game] = int(record[2])
            bd.at['win_percentage', game] = (float(record[2])) / (float(record[2]) + float(record[3]) - 1)

        # No Changes to these things, cuz defense is dumb ugh
        bd.at['srs', game] = float(re.split('[, !?:]', top.find('strong', text='SRS').parent.parent.text)[2])
        bd.at['sos', game] = float(re.split('[, !?:]', top.find('strong', text='SOS').parent.parent.text)[2])
        bd.at['opp_pass_cmp', game] = body.find('td', attrs={'data-stat': 'opp_pass_cmp'}).text
        bd.at['opp_pass_att', game] = body.find('td', attrs={'data-stat': 'opp_pass_att'}).text
        bd.at['opp_pass_cmp_pct', game] = body.find('td', attrs={'data-stat': 'opp_pass_cmp_pct'}).text
        bd.at['opp_pass_yds', game] = body.find('td', attrs={'data-stat': 'opp_pass_yds'}).text
        bd.at['opp_pass_td', game] = body.find('td', attrs={'data-stat': 'opp_pass_td'}).text
        bd.at['opp_tot_yds', game] = body.find('td', attrs={'data-stat': 'opp_tot_yds'}).text
        bd.at['opp_tot_yds_per_play', game] = body.find('td', attrs={'data-stat': 'opp_tot_yds_per_play'}).text
        bd.at['opp_first_down', game] = body.find('td', attrs={'data-stat': 'opp_first_down'}).text
        bd.at['opp_penalty', game] = body.find('td', attrs={'data-stat': 'opp_penalty'}).text
        bd.at['opp_turnovers', game] = body.find('td', attrs={'data-stat': 'opp_turnovers'}).text

        # Average Change needed
        bd.at['ppg', game] = (float(re.split('[, \-!?:]', top.find('strong', text='Points/G:').parent.text)[2]) *
                           total_games_played - float(pf))/(float(total_games_played) - 1)

        bd.at['oppg', game] = (float(top.find('strong', text='Opp Pts/G:').parent.text.split()[2]) *
                           total_games_played - float(pa))/(float(total_games_played) - 1)


        # Changes to be made
        bd.at['pass_cmp', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'pass_cmp'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'pass_cmp'}).text), total_games_played)

        bd.at['pass_att', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'pass_att'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'pass_att'}).text), total_games_played)

        bd.at['pass_cmp_pct', game] = bd[game]['pass_cmp']/bd[game]['pass_att']

        bd.at['pass_yds', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'pass_yds'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'pass_yds'}).text), total_games_played)

        bd.at['pass_td', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'pass_td'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'pass_td'}).text), total_games_played)

        bd.at['rush_att', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'rush_att'}).text),
                                                     float(bg_offense.find('td', attrs={'data-stat': 'pass_att'}).text),
                                                     total_games_played)

        bd.at['rush_yds', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'rush_yds'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'rush_yds'}).text), total_games_played)

        bd.at['rush_yds_per_att', game] = bd[game]['rush_yds'] / bd[game]['rush_att']

        bd.at['rush_td', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'rush_td'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'rush_yds'}).text), total_games_played)

        bd.at['tot_plays', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'tot_plays'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'tot_plays'}).text), total_games_played)

        bd.at['tot_yds', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'tot_yds'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'tot_yds'}).text), total_games_played)

        bd.at['tot_yds_per_play', game] = bd[game]['tot_yds']/bd[game]['tot_plays']

        bd.at['first_down', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'first_down'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'first_down'}).text), total_games_played)

        bd.at['penalty', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'penalty'}).text),
                                                    float(bg_offense.find('td', attrs={'data-stat': 'penalty'}).text),
                                                    total_games_played)

        bd.at['turnovers', game] = adjustAverage(float(body.find('td', attrs={'data-stat': 'turnovers'}).text),
                     float(bg_offense.find('td', attrs={'data-stat': 'turnovers'}).text), total_games_played)

    bdt = bd.T
    bdt.to_csv(filename)

# Main --------------------------------------------------------------
def main():
    link = "https://www.sports-reference.com/cfb/years/2018-schedule.html"
    start_row = 846
    currentData = True
    year = '2018'

    if currentData:
        games, _ = requestBowls([link], [start_row], [year])
        links, teams, team_link, nplinks, npteams, games = parseLink(games)
        filename = '2018_Data.csv'
        createPreviousDataFrame(games, nplinks, npteams, filename)

    links = []
    years = ['2017', '2016', '2015', '2014', '2013']
    lines = [834, 832, 830, 829, 815]
    pastData = False
    if pastData:
        for year in years:
            links.append('https://www.sports-reference.com/cfb/years/' + year + '-schedule.html')

        games_links, years = requestBowls(links, lines, years)
        links, teams, team_link, nplinks, npteams, games, score = parseLinkPrevious(games_links, years)
        for i in range(len(npteams)//2):
            print('Winner: {} {} pts, Loser: {} {} pts, Year: {}'.format(npteams[2*i], score[2*i], npteams[2*i+1],
                                                                         score[2*i+1], years[i]))
        print(len(games_links))
        filename = 'TrialCases.csv'
        createPreviousDataFrame(games, nplinks, npteams, filename)



if __name__ == '__main__':
    main()

