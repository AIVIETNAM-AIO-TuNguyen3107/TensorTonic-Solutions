-- Returns: name, salary.
select name, salary from employees
where 1=1
and salary > 70000
and department in ('Engineering', 'Marketing')
order by salary desc