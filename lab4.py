def compute_infection_likelihood(contact, travel, fever, cough):
    prior_infection = 0.5
    infection_given_contact = {'Yes': 0.90, 'No': 0.10}
    infection_given_travel = {'Yes': 0.80, 'No': 0.20}
    fever_given_infection = 0.85
    cough_given_infection = 0.75

    prior_infection *= infection_given_contact[contact] * infection_given_travel[travel]
    prior_no_infection = (1 - prior_infection) * ((1 - infection_given_contact[contact]) * (1 - infection_given_travel[travel]))

    if fever == 'Yes':
        prior_infection *= fever_given_infection
        prior_no_infection *= (1 - fever_given_infection)
    if cough == 'Yes':
        prior_infection *= cough_given_infection
        prior_no_infection *= (1 - cough_given_infection)

    total_probability = prior_infection + prior_no_infection
    return prior_infection / total_probability, prior_no_infection / total_probability

evaluation_cases = [
    ('Yes', 'Yes', 'Yes', 'Yes'),
    ('No', 'Yes', 'Yes', 'Yes'),
    ('No', 'No', 'No', 'No'),
    ('Yes', 'No', 'Yes', 'No'),
    ('No', 'Yes', 'No', 'No'),
]

for case in evaluation_cases:
    infection_prob, _ = compute_infection_likelihood(*case)
    print(f"Test case {case} -> Infection: {'Yes' if infection_prob > 0.5 else 'No'}, Probability: {infection_prob:.4f}")
