### Tasks

Hi all, this is the outline for our DP library. It includes a Laplace mechanism implementation as an example.

Please use it as a reference for our framework.

Your task for the **next three weeks** is to incorporate *randomize response mechanism, Gaussian mechanism, histogram mechanism and exponential mechanism* to this library. Also, you need to implement some statistical functions using these mechanisms:

- implement `count` function which counts the occurences of value in the column.
- implement `gaussian_mean` function using Gaussian mechanism.
- implement `histogram` function using Laplace mechanism.
- implement `sum/std/var/min/max/median` functions.
- implement `mode` funcion using exponential mechanism.

Note: you will need to create *unit tests* (the more tests, the better) to check your implementation.

Also, we want you to work as a group to implement these functions. Doing so will help you to learn from others. (note: repl.it supports multiplayer mode, see https://repl.it/site/multiplayer).


### Advanced tasks (Optional)

If you want to do more advanced tasks, there are a few:

1. Instead of manually adding and checking privacy budget as in function private_table.py/PrivateTable/update_privacy_loss(). Replace this function by a seperated `SimplePrivacyBudgetTracker` class which does the same job but in OOP fashion.

2. Implement the advanced composition theorem (Theorem 3.20). See Section 3.5.2 and Example 3.7. Add a new  
`AdvancedPrivacyBudgetTracker` class which do the similar job as `SimplePrivacyBudgetTracker` class. Allow the user to select privacy tracker at initialization.

3. (Hard) Implement `RenyiPrivacyBudget` class as a subclass of `PrivacyBudget` and `RenyiPrivacyBudgetTracker` class. See https://www2.cs.duke.edu/courses/fall18/compsci590.1/lectures/10-10-instructor.pdf https://arxiv.org/pdf/1702.07476.pdf

4. (Interesting) Implement the proposed mechanism in this paper https://arxiv.org/abs/2010.12603 and compare the error of this mechanism with the exponential mechanism. 
