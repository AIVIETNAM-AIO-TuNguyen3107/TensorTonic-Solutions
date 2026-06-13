-- Write your SQL query here
select t.department,
    count(*) as total_tickets,
    sum(case when t.status = 'open' then 1 else 0 end) as open_count,
    sum(case when t.status = 'in_progress' then 1 else 0 end) as in_progress_count,
    sum(case when t.status = 'closed' then 1 else 0 end) as closed_count
from tickets as t
group by t.department
ORDER BY total_tickets DESC, department ASC;