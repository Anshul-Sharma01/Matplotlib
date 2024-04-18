from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")

slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java', ]
explode = [0,0,0,0.1,0]







plt.pie(slices,shadow=True,labels=labels,explode=explode,wedgeprops={"edgecolor":"black"},autopct="%1.1f%%",startangle=90)
plt.title("My Awesome Pie Chart")
plt.tight_layout()
plt.show()
