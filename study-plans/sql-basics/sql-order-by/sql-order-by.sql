-- Returns: name, subject, score.
select name, subject, score
from students
order by score desc, name asc nulls last