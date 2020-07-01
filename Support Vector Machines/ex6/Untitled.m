a=[1 2 2; 3 1 2; 6 3 2]
a=a(:)
[p i]=min(a(:))
[l m]=ind2sub(size(a),i)