# ----- Knowledge Base -----
knowledge_base = {
    "facts": [
        "American(West)",
        "Missile(m1)",
        "Enemy(Nono,America)",
        "Owns(Nono,m1)"
    ],
    "rules": [
        ("American(x) and Weapon(y) and Sells(x,y,z) and Enemy(z,America)", "Criminal(x)"),
        ("Missile(m)", "Weapon(m)"),
        ("Missile(m) and Owns(Nono,m)", "Sells(West,m,Nono)")
    ]
}


# ----- Forward Chaining -----
def forward_chain(kb):
    inferred = set(kb["facts"])
    added = True
    while added:
        added = False
        for condition, conclusion in kb["rules"]:
            cond_parts = condition.split(" and ")
            # Check if all parts of condition are matched with known facts
            if all(any(c.split("(")[0] in fact for fact in inferred) for c in cond_parts):
                # Substitute variables with known constants
                conclusion = conclusion.replace("x", "West").replace("y", "m1").replace("z", "Nono")
                if conclusion not in inferred:
                    inferred.add(conclusion)
                    added = True
    return inferred


# ----- Backward Chaining -----
def backward_chain(goal, kb, inferred=None):
    if inferred is None:
        inferred = set(kb["facts"])

    # If goal already known
    if goal in inferred:
        return True

    goal_pred = goal.split("(")[0]

    # Search for rule that can derive this goal
    for condition, conclusion in kb["rules"]:
        if conclusion.split("(")[0] == goal_pred:
            cond_parts = [c.strip() for c in condition.split(" and ")]
            all_true = True
            for cond in cond_parts:
                cond_pred = cond.split("(")[0]
                found = any(cond_pred in f for f in inferred)
                if not found:
                    # recursively prove condition
                    new_goal = cond.replace("x", "West").replace("y", "m1").replace("z", "Nono")
                    if not backward_chain(new_goal, kb, inferred):
                        all_true = False
                        break
            if all_true:
                inferred.add(goal)
                return True
    return False


# ----- Main Execution -----
facts_after_forward = forward_chain(knowledge_base)
print("\n--- Knowledge Base After Forward Chaining ---")
for f in sorted(facts_after_forward):
    print(f"- {f}")

goal = "Criminal(West)"
print("\n--- Backward Chaining Proof for:", goal, "---")
if backward_chain(goal, knowledge_base, set(facts_after_forward)):
    print("✅ Conclusion: West is a Criminal.")
else:
    print("❌ Could not prove the goal.")
