from connect_four_project.game.tournament import Tournament
from connect_four_project.game.connectfour_game import ConnectFour
from connect_four_project.gmc.gradient_monte_carlo import GradientMonteCarlo
import os

players = [GradientMonteCarlo(7, 3e-4), GradientMonteCarlo(7, 3e-4), GradientMonteCarlo(7, 3e-4)]

# load models
if os.path.exists('trained-models'):
    for i, player in enumerate(players):
        try:
            player.value_net.load('trained-models/player ' + str(i))
        except:
            player.value_net.save('trained-models/player ' + str(i))
else:
    os.mkdir('trained-models')
    for i, player in enumerate(players):
        player.value_net.save('trained-models/player ' + str(i))


tournament = Tournament(players, ConnectFour)
# tournament.start_training()
tournament.play_against_human(computer_red=True)