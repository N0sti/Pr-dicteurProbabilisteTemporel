from codeV1 import get_current_datetime, obtenir_donnees_meteo_actuelles_daily, mettre_a_jour_historique_daily

curent_date=get_current_datetime()
data_daily=obtenir_donnees_meteo_actuelles_daily(curent_date)
#print("data_daily:", data_daily)
mettre_a_jour_historique_daily(data_daily)