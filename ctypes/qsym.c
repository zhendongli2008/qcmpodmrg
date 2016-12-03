#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int symAllowed(int rank, int nqnum, int *ndims, int *status, int *allowed,
	       double *a0, double *a1, double *a2, double *a3, 
	       double *a4, double *a5, double *a6, double *a7)
{
   /* maxrank = 10 */
   //printf("rank=%d\n",rank);
   switch (rank){
      case 2: symAllowed2(rank,nqnum,ndims,status,allowed,a0,a1,a2,a3,a4,a5,a6,a7); break;
      case 3: symAllowed3(rank,nqnum,ndims,status,allowed,a0,a1,a2,a3,a4,a5,a6,a7); break;
      case 4: symAllowed4(rank,nqnum,ndims,status,allowed,a0,a1,a2,a3,a4,a5,a6,a7); break;
      case 5: symAllowed5(rank,nqnum,ndims,status,allowed,a0,a1,a2,a3,a4,a5,a6,a7); break;
      case 6: symAllowed6(rank,nqnum,ndims,status,allowed,a0,a1,a2,a3,a4,a5,a6,a7); break;
      case 7: symAllowed7(rank,nqnum,ndims,status,allowed,a0,a1,a2,a3,a4,a5,a6,a7); break;
      case 8: symAllowed8(rank,nqnum,ndims,status,allowed,a0,a1,a2,a3,a4,a5,a6,a7); break;
      default : printf("error\n"); exit(1);
   }
   return 0;
}

int symAllowed2(int rank, int nqnum, int *ndims, int *status, int *allowed,
 	        double *a0, double *a1, double *a2, double *a3, 
	        double *a4, double *a5, double *a6, double *a7)
{
   double *pa[rank];
   double diff[nqnum],dtot;
   int idx,irk,iq;
   int i0,i1,i2,i3,i4,i5,i6,i7;
   //printf("rank 2\n");
   idx  = 0;
   for (i0 = 0; i0 < ndims[0]; i0++) 
   {
      pa[0] = a0 + i0*nqnum;
      for (i1 = 0; i1 < ndims[1]; i1++) 
      {
         pa[1] = a1 + i1*nqnum;
	 //printf("idx=%d : i0=%d, i1=%d\n",idx,i0,i1);
	 // init 
	 for (iq = 0; iq < nqnum; iq++)
	 {
	    diff[iq] = 0.;
	 }
	 // diff
	 for (irk = 0; irk < rank; irk ++)
	 {
	    if (status[irk] == 0)
	    {
	       for (iq = 0; iq < nqnum; iq++)
	       {
		  diff[iq] += pa[irk][iq];
	       }
	    }
	    else {
	       for (iq = 0; iq < nqnum; iq++)
	       {
		  diff[iq] -= pa[irk][iq];
	       }
	    }
	 }
	 // compare
	 dtot = 0.;
	 for (iq = 0; iq < nqnum; iq++)
	 {
	    dtot += fabs(diff[iq]);
	 } 
	 // assign
	 allowed[idx] = dtot < 1.e-10;
	 idx++;
      } // i1
   } // i0
   return 0;
}

int symAllowed3(int rank, int nqnum, int *ndims, int *status, int *allowed,
 	        double *a0, double *a1, double *a2, double *a3, 
	        double *a4, double *a5, double *a6, double *a7)
{
   double *pa[rank];
   double diff[nqnum],dtot;
   int idx,irk,iq;
   int i0,i1,i2,i3,i4,i5,i6,i7;
   //printf("rank 3\n");
   //printf("d1=%d\n",ndims[0]);
   //printf("d2=%d\n",ndims[1]);
   //printf("d3=%d\n",ndims[2]);
   idx  = 0;
   for (i0 = 0; i0 < ndims[0]; i0++) 
   {
      pa[0] = a0 + i0*nqnum;
      for (i1 = 0; i1 < ndims[1]; i1++) 
      {
         pa[1] = a1 + i1*nqnum;
         for (i2 = 0; i2 < ndims[2]; i2++) 
         {
            pa[2] = a2 + i2*nqnum;
	    //printf("idx=%d : i0=%d,i1=%d,i2=%d\n",idx,i0,i1,i2);
	    // init 
	    for (iq = 0; iq < nqnum; iq++)
	    {
	       diff[iq] = 0.;
	    }
	    // diff
	    for (irk = 0; irk < rank; irk ++)
	    {
	       if (status[irk] == 0)
	       {
	          for (iq = 0; iq < nqnum; iq++)
	          {
	             diff[iq] += pa[irk][iq];
	          }
	       }
	       else {
	          for (iq = 0; iq < nqnum; iq++)
	          {
	             diff[iq] -= pa[irk][iq];
	          }
	       }
	    }
	    // compare
	    dtot = 0.;
	    for (iq = 0; iq < nqnum; iq++)
	    {
	       dtot += fabs(diff[iq]);
	    } 
	    // assign
	    allowed[idx] = dtot < 1.e-10;
	    idx++;
	 } // i2
      } // i1
   } // i0
   return 0;
}

int symAllowed4(int rank, int nqnum, int *ndims, int *status, int *allowed,
 	        double *a0, double *a1, double *a2, double *a3, 
	        double *a4, double *a5, double *a6, double *a7)
{
   double *pa[rank];
   double diff[nqnum],dtot;
   int idx,irk,iq;
   int i0,i1,i2,i3,i4,i5,i6,i7;
   //printf("rank 2\n");
   idx  = 0;
   for (i0 = 0; i0 < ndims[0]; i0++) 
   {
      pa[0] = a0 + i0*nqnum;
      for (i1 = 0; i1 < ndims[1]; i1++) 
      {
         pa[1] = a1 + i1*nqnum;
         for (i2 = 0; i2 < ndims[2]; i2++) 
         {
            pa[2] = a2 + i2*nqnum;
            for (i3 = 0; i3 < ndims[3]; i3++) 
            {
               pa[3] = a3 + i3*nqnum;
	       //printf("idx=%d : i0=%d, i1=%d\n",idx,i0,i1);
	       // init 
	       for (iq = 0; iq < nqnum; iq++)
	       {
	          diff[iq] = 0.;
	       }
	       // diff
	       for (irk = 0; irk < rank; irk ++)
	       {
	          if (status[irk] == 0)
	          {
	             for (iq = 0; iq < nqnum; iq++)
	             {
	                diff[iq] += pa[irk][iq];
	             }
	          }
	          else {
	             for (iq = 0; iq < nqnum; iq++)
	             {
	                diff[iq] -= pa[irk][iq];
	             }
	          }
	       }
	       // compare
	       dtot = 0.;
	       for (iq = 0; iq < nqnum; iq++)
	       {
	          dtot += fabs(diff[iq]);
	       } 
	       // assign
	       allowed[idx] = dtot < 1.e-10;
	       idx++;
	    } // i3
	 } // i2
      } // i1
   } // i0
   return 0;
}

int symAllowed5(int rank, int nqnum, int *ndims, int *status, int *allowed,
 	        double *a0, double *a1, double *a2, double *a3, 
	        double *a4, double *a5, double *a6, double *a7)
{
   double *pa[rank];
   double diff[nqnum],dtot;
   int idx,irk,iq;
   int i0,i1,i2,i3,i4,i5,i6,i7;
   //printf("rank 2\n");
   idx  = 0;
   for (i0 = 0; i0 < ndims[0]; i0++) 
   {
      pa[0] = a0 + i0*nqnum;
      for (i1 = 0; i1 < ndims[1]; i1++) 
      {
         pa[1] = a1 + i1*nqnum;
         for (i2 = 0; i2 < ndims[2]; i2++) 
         {
            pa[2] = a2 + i2*nqnum;
            for (i3 = 0; i3 < ndims[3]; i3++) 
            {
               pa[3] = a3 + i3*nqnum;
               for (i4 = 0; i4 < ndims[4]; i4++) 
               {
                  pa[4] = a4 + i4*nqnum;
	          //printf("idx=%d : i0=%d, i1=%d\n",idx,i0,i1);
	          // init 
	          for (iq = 0; iq < nqnum; iq++)
	          {
	             diff[iq] = 0.;
	          }
	          // diff
	          for (irk = 0; irk < rank; irk ++)
	          {
	             if (status[irk] == 0)
	             {
	                for (iq = 0; iq < nqnum; iq++)
	                {
	                   diff[iq] += pa[irk][iq];
	                }
	             }
	             else {
	                for (iq = 0; iq < nqnum; iq++)
	                {
	                   diff[iq] -= pa[irk][iq];
	                }
	             }
	          }
	          // compare
	          dtot = 0.;
	          for (iq = 0; iq < nqnum; iq++)
	          {
	             dtot += fabs(diff[iq]);
	          } 
	          // assign
	          allowed[idx] = dtot < 1.e-10;
	          idx++;
	       } // i4
	    } // i3
	 } // i2
      } // i1
   } // i0
   return 0;
}

int symAllowed6(int rank, int nqnum, int *ndims, int *status, int *allowed,
 	        double *a0, double *a1, double *a2, double *a3, 
	        double *a4, double *a5, double *a6, double *a7)
{
   double *pa[rank];
   double diff[nqnum],dtot;
   int idx,irk,iq;
   int i0,i1,i2,i3,i4,i5,i6,i7;
   //printf("rank 2\n");
   idx  = 0;
   for (i0 = 0; i0 < ndims[0]; i0++) 
   {
      pa[0] = a0 + i0*nqnum;
      for (i1 = 0; i1 < ndims[1]; i1++) 
      {
         pa[1] = a1 + i1*nqnum;
         for (i2 = 0; i2 < ndims[2]; i2++) 
         {
            pa[2] = a2 + i2*nqnum;
            for (i3 = 0; i3 < ndims[3]; i3++) 
            {
               pa[3] = a3 + i3*nqnum;
               for (i4 = 0; i4 < ndims[4]; i4++) 
               {
                  pa[4] = a4 + i4*nqnum;
                  for (i5 = 0; i5 < ndims[5]; i5++) 
                  {
                     pa[5] = a5 + i5*nqnum;
	             //printf("idx=%d : i0=%d, i1=%d\n",idx,i0,i1);
	             // init 
	             for (iq = 0; iq < nqnum; iq++)
	             {
	                diff[iq] = 0.;
	             }
	             // diff
	             for (irk = 0; irk < rank; irk ++)
	             {
	                if (status[irk] == 0)
	                {
	                   for (iq = 0; iq < nqnum; iq++)
	                   {
	                      diff[iq] += pa[irk][iq];
	                   }
	                }
	                else {
	                   for (iq = 0; iq < nqnum; iq++)
	                   {
	                      diff[iq] -= pa[irk][iq];
	                   }
	                }
	             }
	             // compare
	             dtot = 0.;
	             for (iq = 0; iq < nqnum; iq++)
	             {
	                dtot += fabs(diff[iq]);
	             } 
	             // assign
	             allowed[idx] = dtot < 1.e-10;
	             idx++;
	          } // i5
	       } // i4
	    } // i3
	 } // i2
      } // i1
   } // i0
   return 0;
}

int symAllowed7(int rank, int nqnum, int *ndims, int *status, int *allowed,
 	        double *a0, double *a1, double *a2, double *a3, 
	        double *a4, double *a5, double *a6, double *a7)
{
   double *pa[rank];
   double diff[nqnum],dtot;
   int idx,irk,iq;
   int i0,i1,i2,i3,i4,i5,i6,i7;
   //printf("rank 2\n");
   idx  = 0;
   for (i0 = 0; i0 < ndims[0]; i0++) 
   {
      pa[0] = a0 + i0*nqnum;
      for (i1 = 0; i1 < ndims[1]; i1++) 
      {
         pa[1] = a1 + i1*nqnum;
         for (i2 = 0; i2 < ndims[2]; i2++) 
         {
            pa[2] = a2 + i2*nqnum;
            for (i3 = 0; i3 < ndims[3]; i3++) 
            {
               pa[3] = a3 + i3*nqnum;
               for (i4 = 0; i4 < ndims[4]; i4++) 
               {
                  pa[4] = a4 + i4*nqnum;
                  for (i5 = 0; i5 < ndims[5]; i5++) 
                  {
                     pa[5] = a5 + i5*nqnum;
                     for (i6 = 0; i6 < ndims[6]; i6++) 
                     {
                        pa[6] = a6 + i6*nqnum;
	                //printf("idx=%d : i0=%d, i1=%d\n",idx,i0,i1);
	                // init 
	                for (iq = 0; iq < nqnum; iq++)
	                {
	                   diff[iq] = 0.;
	                }
	                // diff
	                for (irk = 0; irk < rank; irk ++)
	                {
	                   if (status[irk] == 0)
	                   {
	                      for (iq = 0; iq < nqnum; iq++)
	                      {
	                         diff[iq] += pa[irk][iq];
	                      }
	                   }
	                   else {
	                      for (iq = 0; iq < nqnum; iq++)
	                      {
	                         diff[iq] -= pa[irk][iq];
	                      }
	                   }
	                }
	                // compare
	                dtot = 0.;
	                for (iq = 0; iq < nqnum; iq++)
	                {
	                   dtot += fabs(diff[iq]);
	                } 
	                // assign
	                allowed[idx] = dtot < 1.e-10;
	                idx++;
	             } // i6
	          } // i5
	       } // i4
	    } // i3
	 } // i2
      } // i1
   } // i0
   return 0;
}

int symAllowed8(int rank, int nqnum, int *ndims, int *status, int *allowed,
 	        double *a0, double *a1, double *a2, double *a3, 
	        double *a4, double *a5, double *a6, double *a7)
{
   double *pa[rank];
   double diff[nqnum],dtot;
   int idx,irk,iq;
   int i0,i1,i2,i3,i4,i5,i6,i7;
   //printf("rank 2\n");
   idx  = 0;
   for (i0 = 0; i0 < ndims[0]; i0++) 
   {
      pa[0] = a0 + i0*nqnum;
      for (i1 = 0; i1 < ndims[1]; i1++) 
      {
         pa[1] = a1 + i1*nqnum;
         for (i2 = 0; i2 < ndims[2]; i2++) 
         {
            pa[2] = a2 + i2*nqnum;
            for (i3 = 0; i3 < ndims[3]; i3++) 
            {
               pa[3] = a3 + i3*nqnum;
               for (i4 = 0; i4 < ndims[4]; i4++) 
               {
                  pa[4] = a4 + i4*nqnum;
                  for (i5 = 0; i5 < ndims[5]; i5++) 
                  {
                     pa[5] = a5 + i5*nqnum;
                     for (i6 = 0; i6 < ndims[6]; i6++) 
                     {
                        pa[6] = a6 + i6*nqnum;
                        for (i7 = 0; i7 < ndims[6]; i7++) 
                        {
                           pa[7] = a7 + i7*nqnum;
	                   //printf("idx=%d : i0=%d, i1=%d\n",idx,i0,i1);
	                   // init 
	                   for (iq = 0; iq < nqnum; iq++)
	                   {
	                      diff[iq] = 0.;
	                   }
	                   // diff
	                   for (irk = 0; irk < rank; irk ++)
	                   {
	                      if (status[irk] == 0)
	                      {
	                         for (iq = 0; iq < nqnum; iq++)
	                         {
	                            diff[iq] += pa[irk][iq];
	                         }
	                      }
	                      else {
	                         for (iq = 0; iq < nqnum; iq++)
	                         {
	                            diff[iq] -= pa[irk][iq];
	                         }
	                      }
	                   }
	                   // compare
	                   dtot = 0.;
	                   for (iq = 0; iq < nqnum; iq++)
	                   {
	                      dtot += fabs(diff[iq]);
	                   } 
	                   // assign
	                   allowed[idx] = dtot < 1.e-10;
	                   idx++;
	                } // i7
	             } // i6
	          } // i5
	       } // i4
	    } // i3
	 } // i2
      } // i1
   } // i0
   return 0;
}
