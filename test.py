import cvxpy as cvx

# Create two scalar optimization variables.
x = cvx.Variable()

# Create two constraints.
constraints = [x + y == 1,
               x - y >= 1]

# Form objective.
obj = cvx.Minimize(cvx.square(x - y))

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)

def SOCP(y, A, Epsilon):
	size = A.shape[1]
	x = cvx.Variable(size)
	obj = norm(x,1)
	obj = cvx.Minimize(obj)

	constraints = [ norm(A * x - y,2) < Epsilon]

	pro = cvx.Problem(obj, constraints)
	prob.solve()

	if prob.status == 'OPTIMAL':
		return x.value
	else
		return -1