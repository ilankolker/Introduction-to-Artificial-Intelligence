import sys


def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file
    "*** YOUR CODE HERE ***"

    domain_file.write("Propositions:\n")
    for disk in disks:
        for peg in pegs:
            domain_file.write(disk + "in" + peg + " ")
            domain_file.write(disk + "not_in" + peg + " ")

    domain_file.write("\nActions:")
    for disk_ind, disk in enumerate(disks):
        for start_peg in pegs:
            for end_peg in pegs:
                if start_peg != end_peg:
                    domain_file.write("\nName: M" + disk + start_peg + end_peg + " ")
                    domain_file.write("\npre: " + disk + "in" + start_peg + " ")
                    for smaller_disk in disks[:disk_ind]:
                        domain_file.write(smaller_disk + "not_in" + start_peg + " ")
                        domain_file.write(smaller_disk + "not_in" + end_peg + " ")
                    domain_file.write("\nadd: " + disk + "in" + end_peg + " " +
                                      disk + "not_in" + start_peg + " ")
                    domain_file.write("\ndelete: " + disk + "in" + start_peg + " " +
                                      disk + "not_in" + end_peg + " ")

    domain_file.close()


def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file
    "*** YOUR CODE HERE ***"

    problem_file.write("Initial state: ")
    for disk in disks:
        problem_file.write(disk + "in" + pegs[0] + " ")
        for peg in pegs[1:]:
            problem_file.write(disk + "not_in" + peg + " ")

    problem_file.write("\nGoal state: ")
    for disk in disks:
        problem_file.write(disk + "in" + pegs[-1] + " ")
        for peg in pegs[:-1]:
            problem_file.write(disk + "not_in" + peg + " ")

    problem_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
