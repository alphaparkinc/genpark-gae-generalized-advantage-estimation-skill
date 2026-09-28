from client import GeneralizedAdvantageEstimator

rewards = [1.0, 1.0, 2.0]
values = [0.5, 1.0, 1.5]
res = GeneralizedAdvantageEstimator.compute_gae(rewards, values, next_value=0.0)

print("Computed GAE Advantages:", [round(a, 4) for a in res['advantages']])
print("Estimated Target Returns:", [round(r, 4) for r in res['returns']])
